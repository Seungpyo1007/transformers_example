from transformers import MarianMTModel, MarianTokenizer

# 모델과 토크나이저를 로드합니다.
model_name = 'Helsinki-NLP/opus-mt-en-ko'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 번역할 텍스트를 정의합니다.
text_to_translate = "Hello, how are you?"

# 텍스트를 번역을 위해 토큰화합니다.
tokenized_text = tokenizer.prepare_seq2seq_batch([text_to_translate], return_tensors="pt")

# 모델을 사용하여 번역합니다.
translated = model.generate(**tokenized_text)

# 번역된 텍스트를 디코딩하여 한국어로 출력합니다.
translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

print(translated_text[0])
