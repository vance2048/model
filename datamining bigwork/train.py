import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import EarlyStoppingCallback
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import re
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn

# 定义自定义数据集类
class ToxicCommentsDataset(Dataset):
    def __init__(self, encodings, labels, toxic_word_features=None):
        self.encodings = encodings
        self.labels = labels
        self.toxic_word_features = toxic_word_features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if self.toxic_word_features is not None:
            item['toxic_word_features'] = torch.tensor(self.toxic_word_features[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 读取训练数据
toxic_comments_df = pd.read_csv('LoL_Toxic_Comments_250.csv')
wordlist_df = pd.read_csv('wordlist_user_mn21lo_bad_lol_20250515072704.csv', skiprows=2)
wordlist_df.columns = wordlist_df.columns.str.replace('"', '').str.strip()
print('wordlist_df columns:', list(wordlist_df.columns))
item_col = [col for col in wordlist_df.columns if 'item' in col.lower()][0]
print('使用的列名:', item_col)
word_sketch_df = pd.read_csv('cleaned_word_sketch_data_final.csv')

# 读取测试数据
test_df = pd.read_csv('LoL_Toxic_Comments_Test.csv')

# 显示数据结构
print("训练集信息：")
print(f"训练集样本数：{len(toxic_comments_df)}")
print(f"训练集标签分布：\n{toxic_comments_df['Label'].value_counts()}")
print("\n测试集信息：")
print(f"测试集样本数：{len(test_df)}")
print(f"测试集标签分布：\n{test_df['Label'].value_counts()}")

# 处理词汇特征
def create_toxic_word_features(texts, wordlist_df, word_sketch_df):
    # 创建毒性词汇集合
    toxic_words = set(wordlist_df[item_col].astype(str).str.lower())
    # 创建词汇搭配特征
    collocate_dict = {}
    for _, row in word_sketch_df.iterrows():
        if pd.notna(row['Keyword']) and pd.notna(row['Collocate']):
            if row['Keyword'] not in collocate_dict:
                collocate_dict[row['Keyword'].lower()] = set()
            collocate_dict[row['Keyword'].lower()].add(row['Collocate'].lower())
    
    features = []
    for text in texts:
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # 计算特征
        toxic_word_count = len(words.intersection(toxic_words))
        collocate_count = 0
        for word in words:
            if word in collocate_dict:
                collocate_count += len(words.intersection(collocate_dict[word]))
        
        # 归一化特征
        total_words = len(words)
        if total_words > 0:
            toxic_word_ratio = toxic_word_count / total_words
            collocate_ratio = collocate_count / total_words
        else:
            toxic_word_ratio = 0
            collocate_ratio = 0
            
        features.append([toxic_word_ratio, collocate_ratio])
    
    return np.array(features)

# 准备训练数据
train_texts = toxic_comments_df['Chat'].tolist()
train_labels = toxic_comments_df['Label'].apply(lambda x: 1 if x == 'Toxic' else 0).tolist()

# 准备测试数据
test_texts = test_df['Chat'].tolist()
test_labels = test_df['Label'].apply(lambda x: 1 if x == 'Toxic' else 0).tolist()

# 生成词汇特征
train_toxic_features = create_toxic_word_features(train_texts, wordlist_df, word_sketch_df)
test_toxic_features = create_toxic_word_features(test_texts, wordlist_df, word_sketch_df)

# 使用BERT的tokenizer进行文本编码
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 编码训练数据
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
train_dataset = ToxicCommentsDataset(train_encodings, train_labels, train_toxic_features)

# 编码测试数据
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
test_dataset = ToxicCommentsDataset(test_encodings, test_labels, test_toxic_features)

# 定义评估函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class BertWithToxicFeatures(BertPreTrainedModel):
    def __init__(self, config, toxic_feature_dim=2):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + toxic_feature_dim, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, toxic_word_features=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]  # [CLS]向量
        if toxic_word_features is not None:
            pooled_output = torch.cat([pooled_output, toxic_word_features.float()], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        output = (logits,)
        return ((loss,) + output) if loss is not None else output

# 替换模型加载部分
model = BertWithToxicFeatures.from_pretrained("bert-base-uncased", num_labels=2, toxic_feature_dim=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    max_steps=50
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # 使用测试集进行评估
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 训练模型
print("\n开始训练模型...")
trainer.train()

# 在测试集上评估模型
print("\n在测试集上评估模型...")
test_results = trainer.evaluate()
print("\n测试集评估结果:")
print(f"准确率: {test_results['eval_accuracy']:.4f}")
print(f"F1分数: {test_results['eval_f1']:.4f}")
print(f"精确率: {test_results['eval_precision']:.4f}")
print(f"召回率: {test_results['eval_recall']:.4f}")

# 获取详细预测结果
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)
true_labels = test_labels

# 输出分类报告
print("\n详细分类报告:")
print(classification_report(true_labels, pred_labels, target_names=['Non-toxic', 'Toxic']))

# 绘制混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-toxic', 'Toxic'],
            yticklabels=['Non-toxic', 'Toxic'])
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig('confusion_matrix.png')
plt.close()

# 保存一些预测示例
print("\n预测示例:")
test_results_df = pd.DataFrame({
    'Text': test_texts,
    'True_Label': ['Toxic' if label == 1 else 'Non-toxic' for label in true_labels],
    'Predicted_Label': ['Toxic' if label == 1 else 'Non-toxic' for label in pred_labels]
})
test_results_df['Correct'] = test_results_df['True_Label'] == test_results_df['Predicted_Label']

# 显示一些错误预测的示例
print("\n错误预测示例:")
print(test_results_df[~test_results_df['Correct']].head())

# 保存完整预测结果
test_results_df.to_csv('test_predictions.csv', index=False)
print("\n预测结果已保存到 test_predictions.csv")
