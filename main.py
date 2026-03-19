# main.py
import re
import math
import random
import asyncio
from typing import List, Dict

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
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
            '"': '"', '《': '》', '（': '）', '(': ')', 
            '[': ']', '{': '}', "'": "'", '【': '】', '<': '>'
        }
        # 定义引用/引号字符
        self.quote_chars = {'"', "'", "`"}

        self.balanced_mode = self.config.get("balanced_split_mode", False)
        try:
            self.min_seg_length = max(int(self.config.get("min_segment_length", 10)), 1)
            self.split_ratio_min = float(self.config.get("balanced_split_ratio_min", 0.4))
            self.split_ratio_max = float(self.config.get("balanced_split_ratio_max", 0.9))
        except (ValueError, TypeError):
            self.min_seg_length = 10
            self.split_ratio_min = 0.4
            self.split_ratio_max = 0.9
            
        self.secondary_pattern = re.compile(r'[，,、；;]+')

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
        标记该事件为 LLM 生成的回复，用于分段作用范围的判定。
        """
        setattr(event, "__is_llm_reply", True)

    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """
        核心拦截器：在消息渲染阶段拦截并执行分段逻辑。
        """
        # 1. 防止重复处理：如果已经处理过则跳过
        if getattr(event, "__splitter_processed", False):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # 2. 作用范围判定：根据配置决定是仅分段 LLM 回复还是分段所有消息
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = getattr(event, "__is_llm_reply", False)
        platform_name = event.get_platform_name()

        if self.config.get("disable_active_split_on_restricted_platforms", True):
            restricted_platforms = {"qq_official", "weixin_official_account", "dingtalk"}
            if platform_name in restricted_platforms:
                return

        if split_scope == "llm_only" and not is_llm_reply:
            return

        # 3. 消息长度校验：如果总长度小于设定阈值，则不执行分段
        max_len_no_split = self.config.get("max_length_no_split", 0)
        total_text_len = sum(len(c.text) for c in result.chain if isinstance(c, Plain))

        if max_len_no_split > 0 and total_text_len < max_len_no_split:
            return

        # 标记为已由分段器处理
        setattr(event, "__splitter_processed", True)

        # === 兼容性处理：脱敏其他插件(如AtTool)注入的零宽字符及附带空格 ===
        # 将其替换为无空格的占位符，避免被带 \s 的分段正则无情切碎导致换行断层
        has_external_at_processing = False
        for comp in result.chain:
            if isinstance(comp, Plain) and comp.text:
                if '\u200b' in comp.text:
                    has_external_at_processing = True
                comp.text = comp.text.replace('\u200b \u200b', '__ZWSP_DOUBLE__')
                comp.text = comp.text.replace('\u200b', '__ZWSP_SINGLE__')

        # 4. 获取分段配置
        split_mode = self.config.get("split_mode", "regex")
        if split_mode == "simple":
            # 简单模式：使用固定字符切分
            split_chars = self.config.get("split_chars", "。？！?!；;\n")
            split_pattern = f"[{re.escape(split_chars)}]+"
        else:
            # 正则模式：使用自定义正则切分
            split_pattern = self.config.get("split_regex", r"[。？！?!\\n…]+")

        clean_pattern = self.config.get("clean_regex", "") # 用于清理无用字符的正则
        smart_mode = self.config.get("enable_smart_split", True) # 是否开启括号保护等智能模式
        max_segs = self.config.get("max_segments", 7) # 最大分段数
        enable_reply = self.config.get("enable_reply", True) # 是否在第一段保留引用回复

        # 非文本组件（图片、艾特、表情等）的分段策略
        strategies = {
            'image': self.config.get("image_strategy", "单独"),
            'at': self.config.get("at_strategy", "跟随下段"),
            'face': self.config.get("face_strategy", "嵌入"),
            'default': self.config.get("other_media_strategy", "跟随下段")
        }

        ideal_length = 0
        if self.balanced_mode and max_segs > 0:
            text_weight = sum(len(c.text.replace(" ", "")) for c in result.chain if isinstance(c, Plain))
            
            solo_media_count = 0
            for c in result.chain:
                if not isinstance(c, Plain) and not isinstance(c, Reply):
                    c_type = type(c).__name__.lower()
                    if 'image' in c_type and strategies.get('image', '单独') == "单独":
                        solo_media_count += 1
                    elif 'at' in c_type and strategies.get('at', '跟随下段') == "单独":
                        solo_media_count += 1
                    elif 'face' in c_type and strategies.get('face', '嵌入') == "单独":
                        solo_media_count += 1
                    elif 'image' not in c_type and 'at' not in c_type and 'face' not in c_type and strategies.get('default', '跟随下段') == "单独":
                        solo_media_count += 1
            
            target_text_segs = max(1, max_segs - solo_media_count)
            
            if text_weight > 0:
                ideal_length = max(math.ceil(text_weight / target_text_segs), self.min_seg_length)
                if text_weight < ideal_length * 1.2:
                    ideal_length = 0

        # 5. 执行切分：将原始消息链分解为多个子链（segments）
        segments = self.split_chain_smart(
            result.chain, 
            split_pattern, 
            smart_mode, 
            strategies, 
            enable_reply,
            ideal_length
        )

        if self.balanced_mode and len(segments) >= 2:
            last_seg_text = "".join([c.text for c in segments[-1] if isinstance(c, Plain)]).replace(" ", "")
            if 0 < len(last_seg_text) < self.min_seg_length:
                last_has_media = any(not isinstance(c, Plain) and not isinstance(c, Reply) for c in segments[-1])
                if not last_has_media:
                    short_tail = segments.pop()
                    segments[-1].extend(short_tail)

        # 6. 最大分段数限制：如果段数过多，则强行合并最后几段，避免消息轰炸
        if len(segments) > max_segs and max_segs > 0:
            logger.warning(f"[Splitter] 分段数({len(segments)}) 超过限制({max_segs})，合并剩余段落。")
            merged_last = []
            final_segments = segments[:max_segs-1]
            for seg in segments[max_segs-1:]:
                merged_last.extend(seg)
            final_segments.append(merged_last)
            segments = final_segments

        # 判定是否需要对 At 组件执行特殊处理逻辑
        at_strategy = strategies.get('at', "跟随下段")
        at_needs_processing = at_strategy in ["接下文", "跟随下段", "嵌入"] and any(
            type(c).__name__.lower() == 'at' for c in result.chain
        )

        # 如果最终只有一段且没有清理需求，则无需分段操作，直接返回由框架处理
        if len(segments) <= 1 and not clean_pattern and not at_needs_processing:
            pass # 这里不能直接 return，需要走到底部进行占位符还原

        # 7. 处理引用回复：仅在第一段注入 Reply 组件
        if enable_reply and segments and event.message_obj.message_id:
            has_reply = any(isinstance(c, Reply) for c in segments[0])
            if not has_reply:
                segments[0].insert(0, Reply(id=event.message_obj.message_id))

        if len(segments) > 1:
            logger.info(f"[Splitter] 消息被分为 {len(segments)} 段。")

        # 8. 预处理：应用清理正则
        if clean_pattern:
            for seg in segments:
                for comp in seg:
                    if isinstance(comp, Plain) and comp.text:
                        comp.text = re.sub(clean_pattern, "", comp.text)

        # 9. 预处理：At 组件前后的空格清理 (智能判断单一单词与长句)
        # 如果已被 AtTool 插件处理过，分段器主动让步，跳过二次清理避免冲突
        if not has_external_at_processing:
            for seg in segments:
                idx = 0
                while idx < len(seg):
                    if type(seg[idx]).__name__.lower() == 'at':
                        # 处理前置空格
                        if at_strategy in ["嵌入", "跟随上段"]:
                            for prev_idx in range(idx - 1, -1, -1):
                                if isinstance(seg[prev_idx], Plain):
                                    text = seg[prev_idx].text
                                    if not re.search(r'[a-zA-Z0-9\']+\s+[a-zA-Z0-9\']+\s+$', text):
                                        seg[prev_idx].text = text.rstrip(" \t")
                                    break
                                elif type(seg[prev_idx]).__name__.lower() not in ['at', 'reply']:
                                    break
                        # 处理后置空格
                        if at_strategy in ["嵌入", "跟随下段", "接下文"]:
                            for next_idx in range(idx + 1, len(seg)):
                                if isinstance(seg[next_idx], Plain):
                                    text = seg[next_idx].text
                                    if not re.search(r'^\s+[a-zA-Z0-9\']+\s+[a-zA-Z0-9\']+', text):
                                        seg[next_idx].text = text.lstrip(" \t")
                                    break
                                elif type(seg[next_idx]).__name__.lower() not in ['at']:
                                    break
                    idx += 1

        # 10. 预处理：注入零宽字符 \u200b，防止 At 后面的文本被误解析
        # 同理，如果已有其他插件注入，则此步主动让步
        if at_needs_processing and not has_external_at_processing:
            for seg in segments:
                idx = 0
                while idx < len(seg):
                    if type(seg[idx]).__name__.lower() == 'at':
                        found_plain = False
                        for next_idx in range(idx + 1, len(seg)):
                            if isinstance(seg[next_idx], Plain):
                                text = seg[next_idx].text
                                if re.search(r'\u200b\s\u200b', text):
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
                    comp.text = comp.text.replace('__ZWSP_DOUBLE__', '\u200b \u200b')
                    comp.text = comp.text.replace('__ZWSP_SINGLE__', '\u200b')

        # 如果只有一段（被上面的判定拦截掉），在这里替换完占位符后直接返回交由框架处理
        if len(segments) <= 1 and not clean_pattern and not at_needs_processing:
            result.chain.clear()
            if segments:
                result.chain.extend(segments[0])
            return

        # 11. 发送前 N-1 段消息
        for i in range(len(segments) - 1):
            segment_chain = segments[i]
            
            # 校验该段落是否为空内容
            text_content = "".join([c.text for c in segment_chain if isinstance(c, Plain)])
            has_media = any(not isinstance(c, Plain) for c in segment_chain)
            if not text_content.strip() and not has_media:
                continue

            try:
                # 尝试为当前分段生成 TTS（语音）组件
                segment_chain = await self._process_tts_for_segment(event, segment_chain)
                
                # 打印日志
                self._log_segment(i + 1, len(segments), segment_chain, "主动发送")

                # 构建消息链并调用上下文接口发送
                mc = MessageChain()
                mc.chain = segment_chain
                await self.context.send_message(event.unified_msg_origin, mc)

                # 计算并执行发送延迟，模拟真人输入感
                wait_time = self.calculate_delay(text_content)
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"[Splitter] 发送分段 {i+1} 失败: {e}")

        # 12. 处理最后一段消息：最后一段直接修改 result.chain，交由框架自身完成最终发送
        if segments:
            last_segment = segments[-1]
            last_text = "".join([c.text for c in last_segment if isinstance(c, Plain)])
            last_has_media = any(not isinstance(c, Plain) for c in last_segment)
            
            if not last_text.strip() and not last_has_media:
                result.chain.clear() # 如果最后一段为空则清除，防止发送空消息
            else:
                self._log_segment(len(segments), len(segments), last_segment, "交给框架")
                result.chain.clear()
                result.chain.extend(last_segment)

    def _log_segment(self, index: int, total: int, chain: List[BaseMessageComponent], method: str):
        """记录分段内容的辅助日志方法"""
        content_str = ""
        for comp in chain:
            if isinstance(comp, Plain):
                content_str += comp.text
            else:
                content_str += f"[{type(comp).__name__}]"
        
        log_content = content_str.replace('\n', '\\n')
        logger.info(f"[Splitter] 第 {index}/{total} 段 ({method}): {log_content}")

    async def _process_tts_for_segment(self, event: AstrMessageEvent, segment: List[BaseMessageComponent]) -> List[BaseMessageComponent]:
        """为单个消息分段转换 TTS 语音"""
        # 检查插件配置是否启用 TTS
        if not self.config.get("enable_tts_for_segments", True):
            return segment
        
        try:
            # 获取当前会话的 TTS 相关全局配置
            all_config = self.context.get_config(event.unified_msg_origin)
            tts_config = all_config.get("provider_tts_settings", {})
            
            # 若框架层面未启用 TTS 则直接返回
            if not tts_config.get("enable", False):
                return segment
            
            # 获取当前正在使用的 TTS 提供商
            tts_provider = self.context.get_using_tts_provider(event.unified_msg_origin)
            if not tts_provider:
                return segment
            
            # 校验是否为 LLM 结果以及是否满足发送语音的条件
            result = event.get_result()
            if not result or not result.is_llm_result():
                return segment
            
            if not await SessionServiceManager.should_process_tts_request(event):
                return segment
            
            # 判定触发概率
            tts_trigger_prob = tts_config.get("trigger_probability", 1.0)
            if random.random() > float(tts_trigger_prob):
                return segment
            
            dual_output = tts_config.get("dual_output", False) # 是否同时发送文本和语音
            
            new_segment = []
            for comp in segment:
                # 仅对长度大于 1 的纯文本进行语音化
                if isinstance(comp, Plain) and len(comp.text) > 1:
                    try:
                        logger.info(f"[Splitter] TTS 请求: {comp.text[:50]}...")
                        audio_path = await tts_provider.get_audio(comp.text)
                        if audio_path:
                            new_segment.append(Record(file=audio_path, url=audio_path))
                            if dual_output: new_segment.append(comp)
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
            # 随机延迟
            return random.uniform(self.config.get("random_min", 1.0), self.config.get("random_max", 3.0))
        elif strategy == "log":
            # 对数延迟：字数越多延迟增加越缓
            base = self.config.get("log_base", 0.5)
            factor = self.config.get("log_factor", 0.8)
            return min(base + factor * math.log(len(text) + 1), 5.0)
        elif strategy == "linear":
            # 线性延迟：固定基础延迟 + (字数 * 系数)
            return self.config.get("linear_base", 0.5) + (len(text) * self.config.get("linear_factor", 0.1))
        else:
            return self.config.get("fixed_delay", 1.5)

    def split_chain_smart(self, chain: List[BaseMessageComponent], pattern: str, smart_mode: bool, strategies: Dict[str, str], enable_reply: bool, ideal_length: int = 0) -> List[List[BaseMessageComponent]]:
        """
        智能分段核心逻辑：遍历消息组件链，根据组件类型和文本内容进行分段。
        """
        segments = []
        current_chain_buffer = []
        current_chain_weight = 0

        for component in chain:
            if isinstance(component, Plain):
                # 文本组件：根据正则模式切分
                text = component.text
                if not text: continue
                if not smart_mode:
                    self._process_text_simple(text, pattern, segments, current_chain_buffer)
                    current_chain_weight = 0
                else:
                    current_chain_weight = self._process_text_smart(
                        text, pattern, segments, current_chain_buffer, current_chain_weight, ideal_length
                    )
            else:
                # 非文本组件（如图片、表情等）
                c_type = type(component).__name__.lower()
                
                # 保留引用组件
                if 'reply' in c_type:
                    if enable_reply: current_chain_buffer.append(component)
                    continue

                # 判定当前组件采用何种分段策略
                if 'image' in c_type: strategy = strategies['image']
                elif 'at' in c_type: strategy = strategies['at']
                elif 'face' in c_type: strategy = strategies['face']
                else: strategy = strategies['default']

                if strategy == "单独":
                    # 将之前的内容打包，当前组件单独作为一段，并开启下一段
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                    segments.append([component])
                    current_chain_weight = 0
                elif strategy == "跟随上段":
                    # 加入当前缓冲区后立即打包分段
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
                    # 立即打包之前的内容，当前组件作为新段落的开头
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                        current_chain_weight = 0
                    current_chain_buffer.append(component)
                else: 
                    # 嵌入模式：直接加入缓冲区，等待自然切分
                    current_chain_buffer.append(component)

        # 收集剩余的消息
        if current_chain_buffer:
            segments.append(current_chain_buffer)
        return [seg for seg in segments if seg]

    def _process_text_simple(self, text: str, pattern: str, segments: list, buffer: list):
        """简单文本切分逻辑"""
        parts = re.split(f"({pattern})", text)
        temp_text = ""
        for part in parts:
            if not part: continue
            if re.fullmatch(pattern, part):
                # 如果匹配到分隔符，则将累计的文本推入分段
                temp_text += part
                buffer.append(Plain(temp_text))
                segments.append(buffer[:])
                buffer.clear()
                temp_text = ""
            else:
                temp_text += part
        if temp_text:
            buffer.append(Plain(temp_text))

    def _process_text_smart(self, text: str, pattern: str, segments: list, buffer: list, start_weight: int = 0, ideal_length: int = 0) -> int:
        """
        智能文本切分逻辑：
        1. 保护代码块（```）。
        2. 保护成对符号（括号、引号）不被中途切断。
        3. 保护英文上下文及数字。
        """
        stack = [] # 符号栈，用于追踪嵌套
        compiled_pattern = re.compile(pattern)
        i = 0
        n = len(text)
        current_chunk = ""
        current_weight = start_weight

        while i < n:
            # 1. 代码块保护：检测到 ``` 则跳过到结束标识，视为一个整体
            if text.startswith("```", i):
                next_idx = text.find("```", i + 3)
                if next_idx != -1:
                    current_chunk += text[i:next_idx+3]
                    current_weight += (next_idx + 3 - i)
                    i = next_idx + 3
                    continue
                else:
                    current_chunk += text[i:]
                    current_weight += (n - i)
                    break

            char = text[i]
            is_opener = char in self.pair_map
            
            # 2. 引号处理
            if char in self.quote_chars:
                if stack and stack[-1] == char: 
                    stack.pop() # 引号闭合
                else: 
                    stack.append(char) # 引号开启
                current_chunk += char
                if not char.isspace(): current_weight += 1
                i += 1
                continue
            
            # 3. 若当前处于成对符号/引号内部
            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer: 
                    stack.pop() # 符号匹配闭合
                elif is_opener and char not in self.quote_chars: 
                    stack.append(char) # 嵌套开启
                
                # 符号内部的换行符替换为空格，保持块的完整性
                if char == '\n': 
                    current_chunk += ' '
                else: 
                    current_chunk += char
                    if not char.isspace(): current_weight += 1
                i += 1
                continue
                
            # 4. 进入新的成对符号
            if is_opener:
                stack.append(char)
                current_chunk += char
                if not char.isspace(): current_weight += 1
                i += 1
                continue

            # 5. 分隔符匹配逻辑
            match = compiled_pattern.match(text, pos=i)
            if match:
                delimiter = match.group()
                should_split = True
                
                if ideal_length > 0 and current_weight < ideal_length * self.split_ratio_min:
                    should_split = False

                prev_char = text[i-1] if i > 0 else ""
                next_char = text[i+len(delimiter)] if i+len(delimiter) < n else ""
                
                # 英文与数字边界保护逻辑
                if '\n' not in delimiter and bool(re.match(r'^[ \t.?!,;:\-\']+$', delimiter)):
                    # 保护英文短句或标点不被切开
                    if bool(re.match(r'^[ \t,;:\-\']+$', delimiter)):
                        prev_is_en = (not prev_char) or bool(re.match(r'^[a-zA-Z0-9 \t.?!,;:\-\']$', prev_char))
                        next_is_en = (not next_char) or bool(re.match(r'^[a-zA-Z0-9 \t.?!,;:\-\']$', next_char))
                        if prev_is_en and next_is_en:
                            current_chunk += delimiter
                            current_weight += len(delimiter)
                            i += len(delimiter)
                            continue
                            
                    # 保护数字中的小数点
                    if bool(re.match(r'^[ \t.?!]+$', delimiter)):
                        if '.' in delimiter and bool(re.match(r'^\d$', prev_char)) and bool(re.match(r'^\d$', next_char)):
                            current_chunk += delimiter
                            current_weight += len(delimiter)
                            i += len(delimiter)
                            continue

                if should_split:
                    # 满足切分条件：将当前块推入缓冲区并打包分段
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
            
            elif ideal_length > 0 and current_weight >= ideal_length * self.split_ratio_max:
                sec_match = self.secondary_pattern.match(text, pos=i)
                if sec_match:
                    delimiter = sec_match.group()
                    is_protected = False

                    if delimiter.strip() in [",", "，", ".", "。"]:
                        prev_char = text[i - 1] if i > 0 else ""
                        next_char = text[i + len(delimiter)] if i + len(delimiter) < n else ""
                        if bool(re.match(r'[a-zA-Z0-9]', prev_char)) and bool(re.match(r'[a-zA-Z0-9]', next_char)):
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
