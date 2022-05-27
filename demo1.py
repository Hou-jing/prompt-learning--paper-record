# step1:任务定义
import torch
from openprompt.data_utils import InputExample
#确定类别，即数据标签
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
#确定数据集
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid = 1,
        text_a = "The film was badly made.",
    ),
]
# step 2 定义预训练模型
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
#step3 模板构建
from openprompt.prompts import ManualTemplate
# prompt采用的手工构建方式，[x],it was [mask].
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
)
#step4 答案映射（答案空间）verbalizer(即把原始标签映射到一组label中）
from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = tokenizer,
)
#step 5 构造Prompt model
from openprompt import PromptForClassification
#PLM,Prompt,Verbalizer是prompt model的主要组成的三个部分
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)
#step 6 prompt dataloader
from openprompt import PromptDataLoader

data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)
# step 7 零样本训练和预测
# making zero-shot inference using pretrained MLM with prompt
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'

wrapped_example = promptTemplate.wrap_one_example(dataset[0])
print(wrapped_example)
'''[[{'text': 'Albert Einstein was one of the greatest intellects of his time.', 'loss_ids': 0, 'shortenable_ids': 1},
 {'text': ' It was', 'loss_ids': 0, 'shortenable_ids': 0}, {'text': '<mask>', 'loss_ids': 1, 'shortenable_ids': 0}], {'guid': 0}]'''