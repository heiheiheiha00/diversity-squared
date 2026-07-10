"""Utilities for locating user-question tokens in multimodal prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class QuestionSpanResult:
    prompt: str
    question_char_spans: List[Tuple[int, int]]
    original_token_positions: List[int]
    expanded_token_positions: List[int]


def image_token_text(model_config, default_image_token: str, default_im_start_token: str, default_im_end_token: str) -> str:
    if getattr(model_config, "mm_use_im_start_end", False):
        return default_im_start_token + default_image_token + default_im_end_token
    return default_image_token


def trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_user_message_and_question_spans(
    question: str,
    *,
    model_config,
    image_placeholder: str,
    default_image_token: str,
    default_im_start_token: str,
    default_im_end_token: str,
) -> Tuple[str, List[Tuple[int, int]]]:
    image_token = image_token_text(model_config, default_image_token, default_im_start_token, default_im_end_token)
    if image_placeholder in question:
        parts = question.split(image_placeholder)
        message = ""
        spans: List[Tuple[int, int]] = []
        for idx, part in enumerate(parts):
            if part:
                start = len(message)
                message += part
                end = len(message)
                trimmed_start, trimmed_end = trim_span(message, start, end)
                if trimmed_end > trimmed_start:
                    spans.append((trimmed_start, trimmed_end))
            if idx < len(parts) - 1:
                message += image_token
        return message, spans

    message = image_token + "\n" + question
    start, end = trim_span(message, len(image_token) + 1, len(message))
    return message, [(start, end)] if end > start else []


def build_llava_prompt_with_question_spans(
    question: str,
    *,
    model_config,
    conv_templates,
    conv_mode: str,
    image_placeholder: str,
    default_image_token: str,
    default_im_start_token: str,
    default_im_end_token: str,
) -> Tuple[str, List[Tuple[int, int]]]:
    if conv_mode not in conv_templates:
        raise KeyError(f"Unknown conv mode {conv_mode!r}.")
    conv = conv_templates[conv_mode].copy()
    user_message, message_spans = build_user_message_and_question_spans(
        question,
        model_config=model_config,
        image_placeholder=image_placeholder,
        default_image_token=default_image_token,
        default_im_start_token=default_im_start_token,
        default_im_end_token=default_im_end_token,
    )
    conv.append_message(conv.roles[0], user_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    message_start = prompt.find(user_message)
    if message_start < 0:
        raise ValueError("Could not locate the user message inside the rendered prompt.")
    return prompt, [(message_start + start, message_start + end) for start, end in message_spans]


def token_count_with_image(prompt: str, tokenizer, tokenizer_image_token, image_token_index: int) -> int:
    return len(tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None))


def char_spans_to_original_token_positions(
    prompt: str,
    spans: Sequence[Tuple[int, int]],
    *,
    tokenizer,
    tokenizer_image_token,
    image_token_index: int,
) -> List[int]:
    positions: List[int] = []
    for start_char, end_char in spans:
        start = token_count_with_image(prompt[:start_char], tokenizer, tokenizer_image_token, image_token_index)
        end = token_count_with_image(prompt[:end_char], tokenizer, tokenizer_image_token, image_token_index)
        positions.extend(range(start, end))
    return sorted(set(int(pos) for pos in positions))


def map_original_positions_to_expanded(
    positions: Iterable[int],
    *,
    image_token_position: int,
    visual_token_count: int,
) -> List[int]:
    expanded: List[int] = []
    shift = int(visual_token_count) - 1
    for pos in positions:
        pos = int(pos)
        if pos < image_token_position:
            expanded.append(pos)
        elif pos > image_token_position:
            expanded.append(pos + shift)
    return sorted(set(expanded))


def llava_question_span_result(
    question: str,
    *,
    model_config,
    tokenizer,
    tokenizer_image_token,
    conv_templates,
    conv_mode: str,
    image_token_index: int,
    image_token_position: int,
    visual_token_count: int,
    image_placeholder: str,
    default_image_token: str,
    default_im_start_token: str,
    default_im_end_token: str,
) -> QuestionSpanResult:
    prompt, char_spans = build_llava_prompt_with_question_spans(
        question,
        model_config=model_config,
        conv_templates=conv_templates,
        conv_mode=conv_mode,
        image_placeholder=image_placeholder,
        default_image_token=default_image_token,
        default_im_start_token=default_im_start_token,
        default_im_end_token=default_im_end_token,
    )
    original_positions = char_spans_to_original_token_positions(
        prompt,
        char_spans,
        tokenizer=tokenizer,
        tokenizer_image_token=tokenizer_image_token,
        image_token_index=image_token_index,
    )
    expanded_positions = map_original_positions_to_expanded(
        original_positions,
        image_token_position=image_token_position,
        visual_token_count=visual_token_count,
    )
    return QuestionSpanResult(
        prompt=prompt,
        question_char_spans=char_spans,
        original_token_positions=original_positions,
        expanded_token_positions=expanded_positions,
    )
