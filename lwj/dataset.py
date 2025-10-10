from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import random
from typing import List
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:# drop dos
            t = t[1:]
        while t[-1] == self.eos_id:# drop eos 
            t = t[:-1]

        if bos and self.bos_id is not None:#if bos isn't null,t.push_front(bos)
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:#if eos isn't null,t.push_back(eos)
            t = t + [self.eos_id]
        return t # 

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class PromptDataset(Dataset):
    def __init__(self, 
                 train_file, 
                 tokenizer, 
                 max_len=2048, 
                 sample=-1, 
                 test = False, 
                 seed=0, 
                 category="", 
                 K=4, 
                 dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if not test:
            if sample > 0:
                self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.K = K
        self.dedup = dedup
        self.instructs = [
            f"",
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    # def generate_example_prompt(self, data_point):
    #     return f"""{data_point["input"]}"""
    
    # def generate_prompt(self, data_point):
    #     return data_point["input"]

    
    def get_history(self, row):
        """
        构造用户历史行为序列的输入输出结构
        将一行数据中的历史商品标题列表拼接为字符串，提取目标商品标题，并判断是否与历史最后一个商品重复。
        Args:
            row (pd.Series): 包含用户历史行为和目标商品信息的一行数据。字段包括：
                - history_item_title (str): 字符串形式的商品标题列表
                - history_item_id (str): 字符串形式的商品 ID 列表
                - item_title (str): 当前目标商品标题
                - item_id (str): 当前目标商品 ID

        Returns:
            dict: 
                - input (str): 拼接后的历史商品标题字符串,""
                - output (str): 当前目标商品标题（带换行符）
                - dedup (bool): 当前商品是否与历史最后一个商品重复
        """
        row['history_item_title'] = eval(row['history_item_title'])# str to list
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += row['history_item_title'][i]
            else:
                history += ", " + row['history_item_title'][i]
        target_item = str(row['item_title'])
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"{history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])#
        target_item = history['output']
        history['output'] = ''
        
        prompt = history["input"]
        tokens = self.tokenizer.encode(prompt, bos=False, eos=False)#encode history_item_title to generate a token 
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "text": prompt,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)#encode target_item and end with a eos
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):#each row
            inputs.append(self.pre(i))
        self.inputs = inputs
    
    
    def get_all_history(self):
        '''
        return all 
        '''
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

