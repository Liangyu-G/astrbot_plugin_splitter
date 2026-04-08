# main.py
import re
import math
import random
import asyncio
from typing import List, Dict

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.message_components import Plain, BaseMessageComponent, Reply, Record
from astrbot.core.star.session_llm_manager import SessionServiceManager


class MessageSplitterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # --- 配置兼容性与迁移逻辑 ---
        self._migrate_config()

        # 定义成对出现的字符，在智能分段时避免在这些符号内部切断
        self.pair_map = {
            '“': '”',
            "《": "》",
            "（": "）",
            "(": ")",
            "[": "]",
            "{": "}",
            "‘": "’",
            "【": "】",
            "<": ">",
        }
        # 定义引用/引号字符
        self.quote_chars = {'"', "'", "`"}

        self.balanced_mode = self.config.get("balanced_split_mode", False)
        try:
            self.min_seg_length = max(int(self.config.get("min_segment_length", 10)), 1)
            self.split_ratio_min = float(
                self.config.get("balanced_split_ratio_min", 0.4)
            )
            self.split_ratio_max = float(
                self.config.get("balanced_split_ratio_max", 0.9)
            )
        except (ValueError, TypeError):
            self.min_seg_length = 10
            self.split_ratio_min = 0.4
            self.split_ratio_max = 0.9

        self.secondary_pattern = re.compile(r"[，,、；;]+")

    def _migrate_config(self):
        """处理旧版本配置数据类型冲突及键名迁移"""
        # 1. 键名迁移: clean_items -> clean_before_items
        if "clean_items" in self.config and "clean_before_items" not in self.config:
            logger.info("[Splitter] 迁移旧配置项 clean_items 至 clean_before_items")
            self.config["clean_before_items"] = self.config.pop("clean_items")

        # 2. 类型强制转换: 将可能为字符串的配置项转换为列表
        list_fields = [
            "split_chars", 
            "clean_before_items", 
            "clean_after_items", 
            "conversation_blacklist", 
            "conversation_whitelist"
        ]
        for field in list_fields:
            val = self.config.get(field)
            if val is not None and isinstance(val, str):
                logger.info(f"[Splitter] 配置项 {field} 类型由 str 迁移至 list")
                if field == "split_chars":
                    self.config[field] = [val] if len(val) > 1 else list(val)
                else:
                    self.config[field] = [val] if val else []
        
        # 3. 确保列表内元素均为字符串
        for field in list_fields:
            val = self.config.get(field)
            if isinstance(val, list):
                self.config[field] = [str(item) for item in val if item is not None]

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.config.get("inject_kaomoji_prompt", True):
            return
        instruction = (
            "\n【特别注意】如果你需要输出颜文字（如 (QAQ)），请务必使用三对反引号包裹，"
            "格式如：```(QAQ)```。这能确保颜文字作为一个整体被发送，不会被分段工具切断。"
        )
        req.system_prompt += instruction

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__is_llm_reply", True)

    def _is_model_generated_reply(self, event: AstrMessageEvent, result) -> bool:
        if not result: return False
        is_model_result = getattr(result, "is_model_result", None)
        if callable(is_model_result):
            try: return bool(is_model_result())
            except: pass
        content_type = getattr(result, "result_content_type", None)
        if content_type is not None:
            type_name = getattr(content_type, "name", "")
            return type_name in {"LLM_RESULT", "AGENT_RUNNER_ERROR", "AGENT_RUNNER_RESULT", "TOOL_RESULT", "TOOL_CALL"}
        return getattr(event, "__is_llm_reply", False)

    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        result = event.get_result()
        if not result or not result.chain: return
        if getattr(result, "__splitter_processed", False): return

        # --- 1. 对话黑白名单校验 ---
        umo = event.unified_msg_origin
        blacklist = self.config.get("conversation_blacklist", [])
        whitelist = self.config.get("conversation_whitelist", [])
        
        if umo in blacklist:
            return
        if whitelist and umo not in whitelist:
            return

        # 群聊开关
        if not self.config.get("enable_group_split", True) and event.message_obj.group_id:
            return

        # 作用范围判定
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = self._is_model_generated_reply(event, result)
        if split_scope == "llm_only" and not is_llm_reply: return

        # --- 2. 长度条件校验 ---
        total_text_len = sum(len(c.text) for c in result.chain if isinstance(c, Plain))
        
        # 最小长度校验（太短不分段）
        max_len_no_split = self.config.get("max_length_no_split", 0)
        if max_len_no_split > 0 and total_text_len < max_len_no_split: return
        
        # 最大长度校验（太长不启用，防止破坏长文本结构）
        max_len_disable = self.config.get("max_length_to_disable", 0)
        if max_len_disable > 0 and total_text_len > max_len_disable:
            logger.debug(f"[Splitter] 文本长度({total_text_len}) 超过禁用阈值({max_len_disable})，跳过分段。")
            return

        setattr(result, "__splitter_processed", True)
        split_mode = self.config.get("split_mode", "regex")

        # --- 分段前清理 ---
        if split_mode == "simple":
            clean_before_items = self.config.get("clean_before_items", [])
            for comp in result.chain:
                if isinstance(comp, Plain) and comp.text:
                    for item in clean_before_items:
                        if item: comp.text = comp.text.replace(item, "")
        else:
            clean_before_regex = self.config.get("clean_before_regex", "")
            if clean_before_regex:
                for comp in result.chain:
                    if isinstance(comp, Plain) and comp.text:
                        comp.text = re.sub(clean_before_regex, "", comp.text, flags=re.DOTALL)

        # 脱敏处理
        has_external_at_processing = False
        for comp in result.chain:
            if isinstance(comp, Plain) and comp.text:
                if "\u200b" in comp.text: has_external_at_processing = True
                comp.text = comp.text.replace("\u200b \u200b", "__ZWSP_DOUBLE__").replace("\u200b", "__ZWSP_SINGLE__")

        # --- 构建分段正则 ---
        if split_mode == "simple":
            split_chars_cfg = self.config.get("split_chars", ["。", "？", "！", "?", "!", "；", ";", "\n"])
            processed_chars = []
            for c in split_chars_cfg:
                if not c: continue
                c_str = str(c).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
                processed_chars.append(re.escape(c_str))
            processed_chars.sort(key=len, reverse=True)
            split_pattern = f"(?:{'|'.join(processed_chars)})+" if processed_chars else r"[\n]+"
        else:
            split_pattern = self.config.get("split_regex", r"[。？！?!\\n…]+")

        smart_mode = self.config.get("enable_smart_split", True) 
        max_segs = self.config.get("max_segments", 7) 
        enable_reply = self.config.get("enable_reply", True) 
        trim_segment_edge_blank_lines = self.config.get("trim_segment_edge_blank_lines", True)

        strategies = {
            "image": self.config.get("image_strategy", "单独"),
            "at": self.config.get("at_strategy", "跟随下段"),
            "face": self.config.get("face_strategy", "嵌入"),
            "default": self.config.get("other_media_strategy", "跟随下段"),
        }

        ideal_length = 0
        if self.balanced_mode and max_segs > 0:
            text_weight = sum(len(c.text.replace(" ", "")) for c in result.chain if isinstance(c, Plain))
            solo_media_count = 0
            for c in result.chain:
                if not isinstance(c, Plain) and not isinstance(c, Reply):
                    c_type = type(c).__name__.lower()
                    s_key = "image" if "image" in c_type else "at" if "at" in c_type else "face" if "face" in c_type else "default"
                    if strategies.get(s_key) == "单独": solo_media_count += 1
            target_text_segs = max(1, max_segs - solo_media_count)
            if text_weight > 0:
                ideal_length = max(math.ceil(text_weight / target_text_segs), self.min_seg_length)
                if text_weight < ideal_length * 1.2: ideal_length = 0

        # 执行切分
        segments = self.split_chain_smart(result.chain, split_pattern, smart_mode, strategies, enable_reply, ideal_length)

        # 尾部合并
        if self.balanced_mode and len(segments) >= 2:
            last_seg_text = "".join([c.text for c in segments[-1] if isinstance(c, Plain)]).replace(" ", "")
            if 0 < len(last_seg_text) < self.min_seg_length:
                if not any(not isinstance(c, (Plain, Reply)) for c in segments[-1]):
                    segments[-2].extend(segments.pop())

        # 最大分段限制
        if len(segments) > max_segs and max_segs > 0:
            final_segments = segments[:max_segs-1]
            merged_last = []
            for seg in segments[max_segs-1:]: merged_last.extend(seg)
            final_segments.append(merged_last)
            segments = final_segments

        # 引用回复处理
        if enable_reply and segments and event.message_obj.message_id:
            if not any(isinstance(c, Reply) for c in segments[0]):
                segments[0].insert(0, Reply(id=event.message_obj.message_id))

        # At 兼容性处理与零宽注入
        at_strategy = strategies.get("at", "跟随下段")
        at_needs_processing = at_strategy in ["接下文", "跟随下段", "嵌入"] and any(type(c).__name__.lower() == "at" for c in result.chain)
        
        if not has_external_at_processing:
            for seg in segments:
                for idx, comp in enumerate(seg):
                    if type(comp).__name__.lower() == "at":
                        if at_strategy in ["嵌入", "跟随上段"]:
                            for p_idx in range(idx-1, -1, -1):
                                if isinstance(seg[p_idx], Plain):
                                    if not re.search(r"[a-zA-Z0-9\']+\s+[a-zA-Z0-9\']+\s+$", seg[p_idx].text):
                                        seg[p_idx].text = seg[p_idx].text.rstrip(" \t")
                                    break
                                elif type(seg[p_idx]).__name__.lower() not in ["at", "reply"]: break
                        if at_strategy in ["嵌入", "跟随下段", "接下文"]:
                            for n_idx in range(idx+1, len(seg)):
                                if isinstance(seg[n_idx], Plain):
                                    if not re.search(r"^\s+[a-zA-Z0-9\']+\s+[a-zA-Z0-9\']+", seg[n_idx].text):
                                        seg[n_idx].text = seg[n_idx].text.lstrip(" \t")
                                    break
                                elif type(seg[n_idx]).__name__.lower() not in ["at"]: break
            if at_needs_processing:
                for seg in segments:
                    idx = 0
                    while idx < len(seg):
                        if type(seg[idx]).__name__.lower() == "at":
                            found = False
                            for n_idx in range(idx+1, len(seg)):
                                if isinstance(seg[n_idx], Plain):
                                    if "\u200b \u200b" not in seg[n_idx].text: seg[n_idx].text = "\u200b \u200b" + seg[n_idx].text
                                    found = True; break
                            if not found: seg.insert(idx+1, Plain("\u200b \u200b"))
                        idx += 1

        # 还原占位符
        for seg in segments:
            for comp in seg:
                if isinstance(comp, Plain) and comp.text:
                    comp.text = comp.text.replace("__ZWSP_DOUBLE__", "\u200b \u200b").replace("__ZWSP_SINGLE__", "\u200b")

        if trim_segment_edge_blank_lines:
            for seg in segments: self._trim_segment_edge_blank_lines(seg)

        # --- 分段后清理 ---
        if split_mode == "simple":
            clean_after_items = self.config.get("clean_after_items", [])
            for seg in segments:
                for comp in seg:
                    if isinstance(comp, Plain) and comp.text:
                        for item in clean_after_items:
                            if item: comp.text = comp.text.replace(item, "")
        else:
            clean_after_regex = self.config.get("clean_after_regex", "")
            if clean_after_regex:
                for seg in segments:
                    for comp in seg:
                        if isinstance(comp, Plain) and comp.text:
                            comp.text = re.sub(clean_after_regex, "", comp.text, flags=re.DOTALL)

        if len(segments) <= 1 and not at_needs_processing:
            result.chain.clear()
            if segments: result.chain.extend(segments[0])
            return

        # 发送前 N-1 段
        for i in range(len(segments) - 1):
            seg_chain = segments[i]
            text_content = "".join([c.text for c in seg_chain if isinstance(c, Plain)])
            if not text_content.strip(" \t\r\n\u200b") and not any(not isinstance(c, Plain) for c in seg_chain):
                continue
            try:
                seg_chain = await self._process_tts_for_segment(event, seg_chain)
                self._log_segment(i + 1, len(segments), seg_chain, "主动发送")
                mc = MessageChain(); mc.chain = seg_chain
                await self.context.send_message(event.unified_msg_origin, mc)
                await asyncio.sleep(self.calculate_delay(text_content))
            except Exception as e:
                logger.error(f"[Splitter] 发送分段 {i+1} 失败: {e}")

        # 最后一段交给框架
        if segments:
            last_seg = segments[-1]
            if not "".join([c.text for c in last_seg if isinstance(c, Plain)]).strip(" \t\r\n\u200b") and not any(not isinstance(c, Plain) for c in last_seg):
                result.chain.clear()
            else:
                self._log_segment(len(segments), len(segments), last_seg, "交给框架")
                result.chain.clear(); result.chain.extend(last_seg)

    def _log_segment(self, index: int, total: int, chain: List[BaseMessageComponent], method: str):
        content = "".join([c.text if isinstance(c, Plain) else f"[{type(c).__name__}]" for c in chain])
        logger.info(f"[Splitter] 第 {index}/{total} 段 ({method}): {content.replace('\n', '\\n')}")

    def _trim_segment_edge_blank_lines(self, segment: List[BaseMessageComponent]) -> None:
        f_p = next((c for c in segment if isinstance(c, Plain)), None)
        l_p = next((c for c in reversed(segment) if isinstance(c, Plain)), None)
        if f_p and f_p.text: f_p.text = re.sub(r'^(?:[ \t]*\r?\n)+', '', f_p.text)
        if l_p and l_p.text: l_p.text = re.sub(r'(?:\r?\n[ \t]*)+$', '', l_p.text)

    async def _process_tts_for_segment(self, event: AstrMessageEvent, segment: List[BaseMessageComponent]) -> List[BaseMessageComponent]:
        if not self.config.get("enable_tts_for_segments", True): return segment
        try:
            all_cfg = self.context.get_config(event.unified_msg_origin)
            tts_cfg = all_cfg.get("provider_tts_settings", {})
            if not tts_cfg.get("enable", False): return segment
            tts_prov = self.context.get_using_tts_provider(event.unified_msg_origin)
            if not tts_prov or not await SessionServiceManager.should_process_tts_request(event): return segment
            if random.random() > float(tts_cfg.get("trigger_probability", 1.0)): return segment
            dual = tts_cfg.get("dual_output", False)
            new_seg = []
            for comp in segment:
                if isinstance(comp, Plain) and len(comp.text) > 1:
                    try:
                        path = await tts_prov.get_audio(comp.text)
                        if path:
                            new_seg.append(Record(file=path, url=path))
                            if dual: new_seg.append(comp)
                        else: new_seg.append(comp)
                    except: new_seg.append(comp)
                else: new_seg.append(comp)
            return new_seg
        except: return segment

    def calculate_delay(self, text: str) -> float:
        strategy = self.config.get("delay_strategy", "linear")
        if strategy == "random": return random.uniform(self.config.get("random_min", 1.0), self.config.get("random_max", 3.0))
        if strategy == "log": return min(self.config.get("log_base", 0.5) + self.config.get("log_factor", 0.8) * math.log(len(text) + 1), 5.0)
        if strategy == "linear": return self.config.get("linear_base", 0.5) + (len(text) * self.config.get("linear_factor", 0.1))
        return self.config.get("fixed_delay", 1.5)

    def split_chain_smart(self, chain: List[BaseMessageComponent], pattern: str, smart: bool, strategies: Dict[str, str], enable_reply: bool, ideal: int = 0) -> List[List[BaseMessageComponent]]:
        segments = []; buffer = []; weight = 0
        for comp in chain:
            if isinstance(comp, Plain):
                if not comp.text: continue
                if not smart: self._process_text_simple(comp.text, pattern, segments, buffer); weight = 0
                else: weight = self._process_text_smart(comp.text, pattern, segments, buffer, weight, ideal)
            else:
                c_type = type(comp).__name__.lower()
                if "reply" in c_type:
                    if enable_reply: buffer.append(comp)
                    continue
                s_key = "image" if "image" in c_type else "at" if "at" in c_type else "face" if "face" in c_type else "default"
                strategy = strategies.get(s_key)
                if strategy == "单独":
                    if buffer: segments.append(buffer[:]); buffer.clear()
                    segments.append([comp]); weight = 0
                elif strategy == "跟随上段":
                    if buffer: buffer.append(comp); segments.append(buffer[:]); buffer.clear(); weight = 0
                    elif segments: segments[-1].append(comp)
                    else: segments.append([comp])
                elif strategy in ["跟随下段", "接下文"]:
                    if buffer: segments.append(buffer[:]); buffer.clear(); weight = 0
                    buffer.append(comp)
                else: buffer.append(comp)
        if buffer: segments.append(buffer)
        return [s for s in segments if s]

    def _process_text_simple(self, text: str, pattern: str, segments: list, buffer: list):
        parts = re.split(f"({pattern})", text)
        tmp = ""
        for p in parts:
            if not p: continue
            if re.fullmatch(pattern, p):
                tmp += p; buffer.append(Plain(tmp))
                segments.append(buffer[:]); buffer.clear(); tmp = ""
            else: tmp += p
        if tmp: buffer.append(Plain(tmp))

    def _process_text_smart(self, text: str, pattern: str, segments: list, buffer: list, start_w: int = 0, ideal: int = 0) -> int:
        stack = []; compiled = re.compile(pattern); i = 0; n = len(text); chunk = ""; weight = start_w
        while i < n:
            if text.startswith("```", i):
                idx = text.find("```", i + 3)
                if idx != -1: chunk += text[i:idx+3]; weight += idx+3-i; i = idx+3; continue
                else: chunk += text[i:]; weight += n-i; break
            if text.startswith("<think>", i):
                idx = text.find("</think>", i + 7)
                if idx != -1: chunk += text[i:idx+8]; weight += idx+8-i; i = idx+8; continue
                else: chunk += text[i:]; weight += n-i; break
            
            match = compiled.match(text, pos=i)
            if match:
                delim = match.group(); should = False
                if not stack or "\n" in delim:
                    should = True
                    if ideal > 0 and weight < ideal * self.split_ratio_min: should = False
                    if should and "\n" not in delim and re.match(r"^[ \t.?!,;:\-\']+$", delim):
                        p_c = text[i-1] if i > 0 else ""; n_c = text[i+len(delim)] if i+len(delim) < n else ""
                        if re.match(r"^[a-zA-Z0-9 \t.?!,;:\-\']$", p_c) and re.match(r"^[a-zA-Z0-9 \t.?!,;:\-\']$", n_c): should = False
                        if "." in delim and p_c.isdigit() and n_c.isdigit(): should = False
                if should:
                    chunk += delim; buffer.append(Plain(chunk))
                    segments.append(buffer[:]); buffer.clear(); chunk = ""; weight = 0; i += len(delim)
                else: chunk += delim; weight += len(delim); i += len(delim)
                continue

            if ideal > 0 and weight >= ideal * self.split_ratio_max and not stack:
                sec = self.secondary_pattern.match(text, pos=i)
                if sec:
                    delim = sec.group(); prot = False
                    if delim.strip() in [",", "，", ".", "。"]:
                        p_c = text[i-1] if i > 0 else ""; n_c = text[i+len(delim)] if i+len(delim) < n else ""
                        if p_c.isalnum() and n_c.isalnum(): prot = True
                    if not prot:
                        chunk += delim; buffer.append(Plain(chunk))
                        segments.append(buffer[:]); buffer.clear(); chunk = ""; weight = 0; i += len(delim)
                        continue

            char = text[i]
            if char in self.quote_chars:
                if stack and stack[-1] == char: stack.pop()
                else: stack.append(char)
                chunk += char; i += 1; weight += 1; continue
            if stack:
                if char == self.pair_map.get(stack[-1]): stack.pop()
                elif char in self.pair_map and char not in self.quote_chars: stack.append(char)
                chunk += char; i += 1; weight += 1; continue
            if char in self.pair_map:
                stack.append(char); chunk += char; i += 1; weight += 1; continue
            chunk += char; i += 1; weight += 1 if not char.isspace() else 0
        if chunk: buffer.append(Plain(chunk))
        return weight
