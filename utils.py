from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    return f"你現在是法官，我會給你一個事件，你必須要給出判決。USER: {instruction} ASSISTANT:"
    '''Format the instruction as a prompt for LLM.'''
#    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"


def get_few_shot_prompt(instruction: str) -> str:
    instruction1 = "翻譯成文言文：\n五年春正月丙午，齊獻武王在晉陽逝世，秘密不公布喪事。"
    answer1 = "五年春正月丙午，齊獻武王薨於晉陽，秘不發喪。"
    instruction2 = "翻譯成現代文：\n祿山構逆，承嗣與張忠誌等為前鋒，陷河洛。\n答案："
    answer2 = "安祿山叛亂，田承嗣和張忠誌等擔任先鋒，攻陷河洛。"
    instruction3 = "翻譯成文言文：\n因此忠貞的臣子，並非不想竭盡忠誠，竭盡忠誠實在太難瞭。"
    answer3 = "故忠貞之臣，非不欲竭誠。竭誠者，乃是極難。"
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。以下是範例：\n 1. USER: {instruction1} ASSISTANT: {answer1}。\n 2. USER: {instruction2} ASSISTANT: {answer2}。\n 3. USER: {instruction3} ASSISTANT: {answer3}。\n 請作答：\n USER: {instruction} ASSISTANT:"
    #return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。以下是範例：\nUSER: {instruction1} ASSISTANT: {answer1}。請回答：\nUSER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:

    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    '''Get the BitsAndBytesConfig.'''
    return config
