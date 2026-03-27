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
        # 定义成对出现的字符，在智能分段时避免在这些符号内部切断
        self.pair_map = {
            '"': '"',
            "《": "》",
            "（": "）",
            "(": ")",
            "[": "]",
            "{": "}",
            "'": "'",
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

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        在大语言模型请求前注入系统提示词。
        引导模型使用特定格式输出颜文字，防止颜文字内部的符号被分段器误识别为切分点。
        """
        if not self.config.get("inject_kaomoji_prompt", True):
            return
        instruction = (
            "\n【特别注意】如果你需要输出颜文字（如 (QAQ)），请务必使用三对反引号包裹，"
            "格式如：```(QAQ)```。这能确保颜文字作为一个整体被发送，不会被分段工具切断。"
        )
        req.system_prompt += instruction

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        为旧版 AstrBot 保留兜底标记。
        新版优先使用 MessageEventResult.result_content_type 判定消息来源。
        """
        setattr(event, "__is_llm_reply", True)

    def _is_model_generated_reply(self, event: AstrMessageEvent, result) -> bool:
        """
        优先使用 AstrBot 本体的结果类型判定消息来源，避免将普通插件结果误判为 LLM 回复。
        仅当运行环境缺少正式类型信息时，才退回旧版事件标记。
        """
        if not result:
            return False

        is_model_result = getattr(result, "is_model_result", None)
        if callable(is_model_result):
            try:
                return bool(is_model_result())
            except Exception as e:
                logger.debug(f"[Splitter] is_model_result() 判定失败，尝试回退: {e}")

        is_llm_result = getattr(result, "is_llm_result", None)
        if callable(is_llm_result):
            try:
                if is_llm_result():
                    return True
            except Exception as e:
                logger.debug(f"[Splitter] is_llm_result() 判定失败，尝试回退: {e}")

        content_type = getattr(result, "result_content_type", None)
        if content_type is not None:
            type_name = getattr(content_type, "name", "")
            # 扩展判断类型：加入工具调用和Agent相关的类型，防止漏切分
            return type_name in {
                "LLM_RESULT",
                "AGENT_RUNNER_ERROR",
                "AGENT_RUNNER_RESULT",
                "TOOL_RESULT",
                "TOOL_CALL"
            }

        return getattr(event, "__is_llm_reply", False)

    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """
        核心拦截器：在消息渲染阶段拦截并执行分段逻辑。
        """
        result = event.get_result()
        if not result or not result.chain:
            return

        # 1. 防止重复处理：将标记绑定在当前结果对象(result)而非事件对象(event)上。
        if getattr(result, "__splitter_processed", False):
            return

        # 2. 群聊分段开关检查 (OneBot11等平台)
        if not self.config.get("enable_group_split", True):
            if event.message_obj.group_id: # 存在 group_id 则为群聊
                return

        # 3. 作用范围判定：根据配置决定是仅分段 LLM 回复还是分段所有消息
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = self._is_model_generated_reply(event, result)

        if split_scope == "llm_only" and not is_llm_reply:
            return

        # 4. 消息长度校验：如果总长度小于设定阈值，则不执行分段
        max_len_no_split = self.config.get("max_length_no_split", 0)
        total_text_len = sum(len(c.text) for c in result.chain if isinstance(c, Plain))

        if max_len_no_split > 0 and total_text_len < max_len_no_split:
            return

        # 标记当前 result 已由分段器处理
        setattr(result, "__splitter_processed", True)

        # 5. 【关键优化】预处理：应用清理项（新手友好）与高级清理正则
        # 将其提前到切分前，保证文本完整性不受切分影响。
        clean_items = self.config.get("clean_items", [])
        clean_pattern = self.config.get("clean_regex", "")
        
        for comp in result.chain:
            if isinstance(comp, Plain) and comp.text:
                # 5.1 先处理简单清理项（列表，精确替换为空）
                if isinstance(clean_items, list):
                    for item in clean_items:
                        if item:
                            comp.text = comp.text.replace(item, "")
                
                # 5.2 再处理高级清理正则
                if clean_pattern:
                    # 使用 re.DOTALL 使得 . 可以匹配换行符
                    comp.text = re.sub(clean_pattern, "", comp.text, flags=re.DOTALL)

        # === 兼容性处理：脱敏其他插件(如AtTool)注入的零宽字符及附带空格 ===
        has_external_at_processing = False
        for comp in result.chain:
            if isinstance(comp, Plain) and comp.text:
                if "\u200b" in comp.text:
                    has_external_at_processing = True
                comp.text = comp.text.replace("\u200b \u200b", "__ZWSP_DOUBLE__")
                comp.text = comp.text.replace("\u200b", "__ZWSP_SINGLE__")

        # 6. 获取分段配置
        split_mode = self.config.get("split_mode", "regex")
        if split_mode == "simple":
            # 简单模式：使用配置的列表动态生成正则表达式
            split_chars_cfg = self.config.get("split_chars", ["。", "？", "！", "?", "!", "；", ";", "\n"])
            if isinstance(split_chars_cfg, str):
                # 兼容旧版本的字符串配置
                split_pattern = f"[{re.escape(split_chars_cfg)}]+"
            else:
                # 处理列表配置，按长度降序排列保证优先匹配长字符串（如 \n\n ）
                escaped_items = [re.escape(str(c)) for c in split_chars_cfg if c]
                escaped_items.sort(key=len, reverse=True)
                if escaped_items:
                    joined = "|".join(escaped_items)
                    split_pattern = f"(?:{joined})+"
                else:
                    split_pattern = r"[\n]+" # 兜底
        else:
            # 正则模式：使用自定义正则切分
            split_pattern = self.config.get("split_regex", r"[。？！?!\\n…]+")

        smart_mode = self.config.get("enable_smart_split", True) # 是否开启括号保护等智能模式
        max_segs = self.config.get("max_segments", 7) # 最大分段数
        enable_reply = self.config.get("enable_reply", True) # 是否在第一段保留引用回复
        trim_segment_edge_blank_lines = self.config.get("trim_segment_edge_blank_lines", True)

        # 非文本组件（图片、艾特、表情等）的分段策略
        strategies = {
            "image": self.config.get("image_strategy", "单独"),
            "at": self.config.get("at_strategy", "跟随下段"),
            "face": self.config.get("face_strategy", "嵌入"),
            "default": self.config.get("other_media_strategy", "跟随下段"),
        }

        ideal_length = 0
        if self.balanced_mode and max_segs > 0:
            text_weight = sum(
                len(c.text.replace(" ", ""))
                for c in result.chain
                if isinstance(c, Plain)
            )

            solo_media_count = 0
            for c in result.chain:
                if not isinstance(c, Plain) and not isinstance(c, Reply):
                    c_type = type(c).__name__.lower()
                    if "image" in c_type and strategies.get("image", "单独") == "单独":
                        solo_media_count += 1
                    elif "at" in c_type and strategies.get("at", "跟随下段") == "单独":
                        solo_media_count += 1
                    elif "face" in c_type and strategies.get("face", "嵌入") == "单独":
                        solo_media_count += 1
                    elif (
                        "image" not in c_type
                        and "at" not in c_type
                        and "face" not in c_type
                        and strategies.get("default", "跟随下段") == "单独"
                    ):
                        solo_media_count += 1

            target_text_segs = max(1, max_segs - solo_media_count)

            if text_weight > 0:
                ideal_length = max(
                    math.ceil(text_weight / target_text_segs), self.min_seg_length
                )
                if text_weight < ideal_length * 1.2:
                    ideal_length = 0

        # 7. 执行切分：将原始消息链分解为多个子链（segments）
        segments = self.split_chain_smart(
            result.chain,
            split_pattern,
            smart_mode,
            strategies,
            enable_reply,
            ideal_length,
        )

        if self.balanced_mode and len(segments) >= 2:
            last_seg_text = "".join(
                [c.text for c in segments[-1] if isinstance(c, Plain)]
            ).replace(" ", "")
            if 0 < len(last_seg_text) < self.min_seg_length:
                last_has_media = any(
                    not isinstance(c, Plain) and not isinstance(c, Reply)
                    for c in segments[-1]
                )
                if not last_has_media:
                    short_tail = segments.pop()
                    segments[-1].extend(short_tail)

        # 8. 最大分段数限制
        if len(segments) > max_segs and max_segs > 0:
            logger.warning(
                f"[Splitter] 分段数({len(segments)}) 超过限制({max_segs})，合并剩余段落。"
            )
            merged_last = []
            final_segments = segments[: max_segs - 1]
            for seg in segments[max_segs - 1 :]:
                merged_last.extend(seg)
            final_segments.append(merged_last)
            segments = final_segments

        # 判定是否需要对 At 组件执行特殊处理逻辑
        at_strategy = strategies.get("at", "跟随下段")
        at_needs_processing = at_strategy in ["接下文", "跟随下段", "嵌入"] and any(
            type(c).__name__.lower() == "at" for c in result.chain
        )

        # 如果最终只有一段且没有清理需求，则无需分段操作
        if len(segments) <= 1 and not clean_pattern and not clean_items and not at_needs_processing:
            pass 

        # 9. 处理引用回复：仅在第一段注入 Reply 组件
        if enable_reply and segments and event.message_obj.message_id:
            has_reply = any(isinstance(c, Reply) for c in segments[0])
            if not has_reply:
                segments[0].insert(0, Reply(id=event.message_obj.message_id))

        if len(segments) > 1:
            logger.info(f"[Splitter] 消息被分为 {len(segments)} 段。")

        # 10. 预处理：At 组件前后的空格清理
        if not has_external_at_processing:
            for seg in segments:
                idx = 0
                while idx < len(seg):
                    if type(seg[idx]).__name__.lower() == "at":
                        # 处理前置空格
                        if at_strategy in ["嵌入", "跟随上段"]:
                            for prev_idx in range(idx - 1, -1, -1):
                                if isinstance(seg[prev_idx], Plain):
                                    text = seg[prev_idx].text
                                    if not re.search(
                                        r"[a-zA-Z0-9\']+\s+[a-zA-Z0-9\']+\s+$", text
                                    ):
                                        seg[prev_idx].text = text.rstrip(" \t")
                                    break
                                elif type(seg[prev_idx]).__name__.lower() not in [
                                    "at",
                                    "reply",
                                ]:
                                    break
                        # 处理后置空格
                        if at_strategy in ["嵌入", "跟随下段", "接下文"]:
                            for next_idx in range(idx + 1, len(seg)):
                                if isinstance(seg[next_idx], Plain):
                                    text = seg[next_idx].text
                                    if not re.search(
                                        r"^\s+[a-zA-Z0-9\']+\s+[a-zA-Z0-9\']+", text
                                    ):
                                        seg[next_idx].text = text.lstrip(" \t")
                                    break
                                elif type(seg[next_idx]).__name__.lower() not in ["at"]:
                                    break
                    idx += 1

        # 11. 预处理：注入零宽字符 \u200b
        if at_needs_processing and not has_external_at_processing:
            for seg in segments:
                idx = 0
                while idx < len(seg):
                    if type(seg[idx]).__name__.lower() == "at":
                        found_plain = False
                        for next_idx in range(idx + 1, len(seg)):
                            if isinstance(seg[next_idx], Plain):
                                text = seg[next_idx].text
                                if re.search(r"\u200b\s\u200b", text):
                                    pass
                                else:
                                    seg[next_idx].text = "\u200b \u200b" + text
                                found_plain = True
                                break
                        if not found_plain:
                            seg.insert(idx + 1, Plain("\u200b \u200b"))
                    idx += 1

        # === 兼容性处理：完美还原脱敏的占位符 ===
        for seg in segments:
            for comp in seg:
                if isinstance(comp, Plain) and comp.text:
                    comp.text = comp.text.replace("__ZWSP_DOUBLE__", "\u200b \u200b")
                    comp.text = comp.text.replace("__ZWSP_SINGLE__", "\u200b")

        # 仅裁剪每段首尾的空白行，保留段内原始换行格式
        if trim_segment_edge_blank_lines:
            for seg in segments:
                self._trim_segment_edge_blank_lines(seg)

        # 如果只有一段且没做处理，替换完占位符直接交给框架
        if len(segments) <= 1 and not clean_pattern and not clean_items and not at_needs_processing:
            result.chain.clear()
            if segments:
                result.chain.extend(segments[0])
            return

        # 12. 发送前 N-1 段消息
        for i in range(len(segments) - 1):
            segment_chain = segments[i]

            text_content = "".join(
                [c.text for c in segment_chain if isinstance(c, Plain)]
            )
            text_for_check = text_content.strip(" \t\r\n\u200b")
            has_media = any(not isinstance(c, Plain) for c in segment_chain)
            if not text_for_check and not has_media:
                continue

            try:
                segment_chain = await self._process_tts_for_segment(
                    event, segment_chain
                )

                self._log_segment(i + 1, len(segments), segment_chain, "主动发送")

                mc = MessageChain()
                mc.chain = segment_chain
                await self.context.send_message(event.unified_msg_origin, mc)

                wait_time = self.calculate_delay(text_content)
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"[Splitter] 发送分段 {i + 1} 失败: {e}")

        # 13. 处理最后一段消息
        if segments:
            last_segment = segments[-1]
            last_text = "".join([c.text for c in last_segment if isinstance(c, Plain)])
            last_text_for_check = last_text.strip(" \t\r\n\u200b")
            last_has_media = any(not isinstance(c, Plain) for c in last_segment)

            if not last_text_for_check and not last_has_media:
                result.chain.clear()
            else:
                self._log_segment(
                    len(segments), len(segments), last_segment, "交给框架"
                )
                result.chain.clear()
                result.chain.extend(last_segment)

    def _log_segment(
        self, index: int, total: int, chain: List[BaseMessageComponent], method: str
    ):
        """记录分段内容的辅助日志方法"""
        content_str = ""
        for comp in chain:
            if isinstance(comp, Plain):
                content_str += comp.text
            else:
                content_str += f"[{type(comp).__name__}]"

        log_content = content_str.replace("\n", "\\n")
        logger.info(f"[Splitter] 第 {index}/{total} 段 ({method}): {log_content}")

    def _trim_segment_edge_blank_lines(self, segment: List[BaseMessageComponent]) -> None:
        """只移除单个分段首尾 Plain 文本中的空白行，保留中间正文换行。"""
        first_plain = None
        last_plain = None

        for comp in segment:
            if isinstance(comp, Plain):
                first_plain = comp
                break

        for comp in reversed(segment):
            if isinstance(comp, Plain):
                last_plain = comp
                break

        if first_plain and first_plain.text:
            first_plain.text = re.sub(r'^(?:[ \t]*\r?\n)+', '', first_plain.text)

        if last_plain and last_plain.text:
            last_plain.text = re.sub(r'(?:\r?\n[ \t]*)+$', '', last_plain.text)

    async def _process_tts_for_segment(self, event: AstrMessageEvent, segment: List[BaseMessageComponent]) -> List[BaseMessageComponent]:
        """为单个消息分段转换 TTS 语音"""
        if not self.config.get("enable_tts_for_segments", True):
            return segment

        try:
            all_config = self.context.get_config(event.unified_msg_origin)
            tts_config = all_config.get("provider_tts_settings", {})

            if not tts_config.get("enable", False):
                return segment

            tts_provider = self.context.get_using_tts_provider(event.unified_msg_origin)
            if not tts_provider:
                return segment

            result = event.get_result()
            if not result or not result.is_llm_result():
                return segment

            if not await SessionServiceManager.should_process_tts_request(event):
                return segment

            tts_trigger_prob = tts_config.get("trigger_probability", 1.0)
            if random.random() > float(tts_trigger_prob):
                return segment

            dual_output = tts_config.get("dual_output", False) 

            new_segment = []
            for comp in segment:
                if isinstance(comp, Plain) and len(comp.text) > 1:
                    try:
                        logger.info(f"[Splitter] TTS 请求: {comp.text[:50]}...")
                        audio_path = await tts_provider.get_audio(comp.text)
                        if audio_path:
                            new_segment.append(Record(file=audio_path, url=audio_path))
                            if dual_output:
                                new_segment.append(comp)
                        else:
                            new_segment.append(comp)
                    except Exception as e:
                        logger.error(f"[Splitter] TTS 处理失败: {e}")
                        new_segment.append(comp)
                else:
                    new_segment.append(comp)
            return new_segment

        except Exception as e:
            logger.error(f"[Splitter] TTS 配置检查失败: {e}")
            return segment

    def calculate_delay(self, text: str) -> float:
        """根据文本长度或策略计算发送间隔延迟"""
        strategy = self.config.get("delay_strategy", "linear")
        if strategy == "random":
            return random.uniform(
                self.config.get("random_min", 1.0), self.config.get("random_max", 3.0)
            )
        elif strategy == "log":
            base = self.config.get("log_base", 0.5)
            factor = self.config.get("log_factor", 0.8)
            return min(base + factor * math.log(len(text) + 1), 5.0)
        elif strategy == "linear":
            return self.config.get("linear_base", 0.5) + (
                len(text) * self.config.get("linear_factor", 0.1)
            )
        else:
            return self.config.get("fixed_delay", 1.5)

    def split_chain_smart(
        self,
        chain: List[BaseMessageComponent],
        pattern: str,
        smart_mode: bool,
        strategies: Dict[str, str],
        enable_reply: bool,
        ideal_length: int = 0,
    ) -> List[List[BaseMessageComponent]]:
        """
        智能分段核心逻辑：遍历消息组件链，根据组件类型和文本内容进行分段。
        """
        segments = []
        current_chain_buffer = []
        current_chain_weight = 0

        for component in chain:
            if isinstance(component, Plain):
                text = component.text
                if not text:
                    continue
                if not smart_mode:
                    self._process_text_simple(
                        text, pattern, segments, current_chain_buffer
                    )
                    current_chain_weight = 0
                else:
                    current_chain_weight = self._process_text_smart(
                        text,
                        pattern,
                        segments,
                        current_chain_buffer,
                        current_chain_weight,
                        ideal_length,
                    )
            else:
                c_type = type(component).__name__.lower()

                if "reply" in c_type:
                    if enable_reply:
                        current_chain_buffer.append(component)
                    continue

                if "image" in c_type:
                    strategy = strategies["image"]
                elif "at" in c_type:
                    strategy = strategies["at"]
                elif "face" in c_type:
                    strategy = strategies["face"]
                else:
                    strategy = strategies["default"]

                if strategy == "单独":
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                    segments.append([component])
                    current_chain_weight = 0
                elif strategy == "跟随上段":
                    if current_chain_buffer:
                        current_chain_buffer.append(component)
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                        current_chain_weight = 0
                    elif segments:
                        segments[-1].append(component)
                    else:
                        segments.append([component])
                elif strategy in ["跟随下段", "接下文"]:
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                        current_chain_weight = 0
                    current_chain_buffer.append(component)
                else:
                    current_chain_buffer.append(component)

        if current_chain_buffer:
            segments.append(current_chain_buffer)
        return [seg for seg in segments if seg]

    def _process_text_simple(
        self, text: str, pattern: str, segments: list, buffer: list
    ):
        """简单文本切分逻辑"""
        parts = re.split(f"({pattern})", text)
        temp_text = ""
        for part in parts:
            if not part:
                continue
            if re.fullmatch(pattern, part):
                temp_text += part
                buffer.append(Plain(temp_text))
                segments.append(buffer[:])
                buffer.clear()
                temp_text = ""
            else:
                temp_text += part
        if temp_text:
            buffer.append(Plain(temp_text))

    def _process_text_smart(
        self,
        text: str,
        pattern: str,
        segments: list,
        buffer: list,
        start_weight: int = 0,
        ideal_length: int = 0,
    ) -> int:
        """
        智能文本切分逻辑
        """
        stack = []  
        compiled_pattern = re.compile(pattern)
        i = 0
        n = len(text)
        current_chunk = ""
        current_weight = start_weight

        while i < n:
            # 1. 代码块保护
            if text.startswith("```", i):
                next_idx = text.find("```", i + 3)
                if next_idx != -1:
                    current_chunk += text[i : next_idx + 3]
                    current_weight += next_idx + 3 - i
                    i = next_idx + 3
                    continue
                else:
                    current_chunk += text[i:]
                    current_weight += n - i
                    break
                    
            # 1.5 XML标签保护：保护 <think> 等思维链标签不被切断
            if text.startswith("<think>", i):
                next_idx = text.find("</think>", i + 7)
                if next_idx != -1:
                    current_chunk += text[i : next_idx + 8]
                    current_weight += next_idx + 8 - i
                    i = next_idx + 8
                    continue
                else:
                    current_chunk += text[i:]
                    current_weight += n - i
                    break

            char = text[i]
            is_opener = char in self.pair_map

            # 2. 引号处理
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    stack.pop()  
                else:
                    stack.append(char)  
                current_chunk += char
                if not char.isspace():
                    current_weight += 1
                i += 1
                continue

            # 3. 若当前处于成对符号/引号内部
            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer:
                    stack.pop()  
                elif is_opener and char not in self.quote_chars:
                    stack.append(char)  

                # 保持符号内部原始字符（包含换行符）不被替换为空格
                current_chunk += char
                if not char.isspace():
                    current_weight += 1
                i += 1
                continue

            # 4. 进入新的成对符号
            if is_opener:
                stack.append(char)
                current_chunk += char
                if not char.isspace():
                    current_weight += 1
                i += 1
                continue

            # 5. 分隔符匹配逻辑
            match = compiled_pattern.match(text, pos=i)
            if match:
                delimiter = match.group()
                should_split = True

                if (
                    ideal_length > 0
                    and current_weight < ideal_length * self.split_ratio_min
                ):
                    should_split = False

                prev_char = text[i - 1] if i > 0 else ""
                next_char = text[i + len(delimiter)] if i + len(delimiter) < n else ""

                if "\n" not in delimiter and bool(
                    re.match(r"^[ \t.?!,;:\-\']+$", delimiter)
                ):
                    if bool(re.match(r"^[ \t,;:\-\']+$", delimiter)):
                        prev_is_en = (not prev_char) or bool(
                            re.match(r"^[a-zA-Z0-9 \t.?!,;:\-\']$", prev_char)
                        )
                        next_is_en = (not next_char) or bool(
                            re.match(r"^[a-zA-Z0-9 \t.?!,;:\-\']$", next_char)
                        )
                        if prev_is_en and next_is_en:
                            current_chunk += delimiter
                            current_weight += len(delimiter)
                            i += len(delimiter)
                            continue

                    if bool(re.match(r"^[ \t.?!]+$", delimiter)):
                        if (
                            "." in delimiter
                            and bool(re.match(r"^\d$", prev_char))
                            and bool(re.match(r"^\d$", next_char))
                        ):
                            current_chunk += delimiter
                            current_weight += len(delimiter)
                            i += len(delimiter)
                            continue

                if should_split:
                    current_chunk += delimiter
                    buffer.append(Plain(current_chunk))
                    segments.append(buffer[:])
                    buffer.clear()
                    current_chunk = ""
                    current_weight = 0
                    i += len(delimiter)
                else:
                    current_chunk += delimiter
                    current_weight += len(delimiter)
                    i += len(delimiter)
                continue

            elif (
                ideal_length > 0
                and current_weight >= ideal_length * self.split_ratio_max
            ):
                sec_match = self.secondary_pattern.match(text, pos=i)
                if sec_match:
                    delimiter = sec_match.group()
                    is_protected = False

                    if delimiter.strip() in [",", "，", ".", "。"]:
                        prev_char = text[i - 1] if i > 0 else ""
                        next_char = (
                            text[i + len(delimiter)] if i + len(delimiter) < n else ""
                        )
                        if bool(re.match(r"[a-zA-Z0-9]", prev_char)) and bool(
                            re.match(r"[a-zA-Z0-9]", next_char)
                        ):
                            is_protected = True

                    if not is_protected:
                        current_chunk += delimiter
                        buffer.append(Plain(current_chunk))
                        segments.append(buffer[:])
                        buffer.clear()
                        current_chunk = ""
                        current_weight = 0
                        i += len(delimiter)
                        continue

            # 常规字符累加
            current_chunk += char
            if not char.isspace():
                current_weight += 1
            i += 1

        if current_chunk:
            buffer.append(Plain(current_chunk))

        return current_weight
