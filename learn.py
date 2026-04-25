# download_model.py
from transformers import BertForSequenceClassification, AutoTokenizer

model_name = 'smilegate-ai/kor_unsmile'
save_path = "./model_files"

# 모델과 토크나이저를 로컬 폴더에 저장
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)