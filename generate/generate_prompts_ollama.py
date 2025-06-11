#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import time
import argparse
import requests
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pypinyin
import emoji
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import subprocess  # 添加这行导入

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



@dataclass
class GenerationConfig:
    """生成配置类"""
    model: str = "qwen3:1.7b"  # Ollama模型名称
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 500
    templates_path: str = "prompt_templates.json"
    topics_path: str = "topics.json"
    output_dir: str = "generated_texts"
    # 不再需要api_url
    apply_adversarial: bool = True
    batch_size: int = 5
    timeout: int = 60  # 添加命令行超时时间
    

class TextGenerator:
    """文本生成器类"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.templates = self._load_json(config.templates_path)
        self.topics = self._load_json(config.topics_path)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 对抗性转换方法映射
        self.adversarial_transforms = {
            "pinyin_mix": self._transform_pinyin_mix,
            "emoji_decorate": self._transform_emoji_decorate,
            "homophone_symbols": self._transform_homophone_symbols,
            "multilingual_mix": self._transform_multilingual_mix
        }
    
    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载{path}失败: {e}")
            raise
    
    def generate_texts(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """生成多个文本样本"""
        results = []
        template_batches = [self.templates[i:i+self.config.batch_size] 
                           for i in range(0, len(self.templates), self.config.batch_size)]
        
        for batch in tqdm(template_batches, desc="处理模板批次"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(self._generate_for_template, template, num_samples // len(self.templates)) 
                          for template in batch]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                    except Exception as e:
                        logger.error(f"生成过程出错: {e}")
        
        # 保存结果
        self._save_results(results)
        return results
    
    def _generate_for_template(self, template: Dict[str, Any], num_samples: int) -> List[Dict[str, Any]]:
        """为特定模板生成多个样本"""
        results = []
        for _ in range(num_samples):
            # 随机选择话题/事件/群体
            topic_data = random.choice(self.topics)
            
            # 准备提示词
            prompt = self._prepare_prompt(template, topic_data)
            
            # 调用模型生成文本
            try:
                generated_text = self._call_ollama(prompt)
                
                # 应用对抗性转换
                adversarial_texts = {}
                if self.config.apply_adversarial and "<content>" not in template["template"]:
                    for transform_name, transform_func in self.adversarial_transforms.items():
                        adversarial_texts[transform_name] = transform_func(generated_text)
                
                # 记录结果
                result = {
                    "template_id": template["id"],
                    "template_description": template["description"],
                    "prompt": prompt,
                    "original_text": generated_text,
                    "adversarial_texts": adversarial_texts,
                    "topic": topic_data.get("topic", ""),
                    "event": topic_data.get("event", ""),
                    "group": topic_data.get("group", ""),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"生成文本时出错，模板ID: {template['id']}, 错误: {e}")
        
        return results
    
    def _prepare_prompt(self, template: Dict[str, Any], topic_data: Dict[str, str]) -> str:
        """准备提示词"""
        prompt_template = template["template"]
        
        # 替换标记
        prompt = prompt_template
        if "<topic>" in prompt and "topic" in topic_data:
            prompt = prompt.replace("<topic>", topic_data["topic"])
        
        if "<event>" in prompt and "event" in topic_data:
            prompt = prompt.replace("<event>", topic_data["event"])
            
        if "<hot_event>" in prompt and "event" in topic_data:
            prompt = prompt.replace("<hot_event>", topic_data["event"])
            
        if "<group>" in prompt and "group" in topic_data:
            prompt = prompt.replace("<group>", topic_data["group"])
            
        if "<content>" in prompt:
            # 如果需要内容填充，先生成原始内容
            content_prompt = f"Write a short controversial comment about {topic_data.get('topic', 'current events')}"
            content = self._call_ollama(content_prompt)
            prompt = prompt.replace("<content>", content)
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """使用命令行调用Ollama模型"""
        try:
            # 使用subprocess调用ollama命令行工具
            result = subprocess.run(
                ["ollama", "run", self.config.model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                # 返回模型输出结果
                return result.stdout.decode("utf-8").strip()
            else:
                # 如果命令行执行失败，抛出异常
                stderr_output = result.stderr.decode("utf-8")
                raise Exception(f"Ollama命令执行失败: {stderr_output}")
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama命令执行超时")
            raise
        except Exception as e:
            logger.error(f"调用Ollama出错: {e}")
            raise
    
    def _transform_pinyin_mix(self, text: str) -> str:
        """拼音混写转换"""
        words = list(text)
        # 随机选择30%的汉字转为拼音
        for i in range(len(words)):
            if re.match(r'[\u4e00-\u9fff]', words[i]) and random.random() < 0.3:
                pinyin_list = pypinyin.lazy_pinyin([words[i]])
                if pinyin_list:
                    words[i] = pinyin_list[0]
        
        return ''.join(words)
    
    def _transform_emoji_decorate(self, text: str) -> str:
        """添加emoji装饰"""
        common_emojis = ['😊', '😂', '🤣', '❤️', '👍', '🔥', '✨', '🙏', '👀', '💯', 
                        '🤔', '😱', '🙄', '😒', '🎉', '💪', '🌹', '🌈', '🤷‍♀️', '🤦‍♂️']
        
        words = list(text)
        result = []
        
        # 在词语间随机插入emoji
        for i, char in enumerate(words):
            result.append(char)
            # 在句子结尾或标点后更可能添加emoji
            if char in ['。', '！', '？', '…', '.', '!', '?', '\n'] and random.random() < 0.7:
                result.append(random.choice(common_emojis))
            # 其他位置小概率添加emoji
            elif random.random() < 0.1:
                result.append(random.choice(common_emojis))
        
        # 开头和结尾一定添加emoji
        result.insert(0, random.choice(common_emojis))
        result.append(random.choice(common_emojis))
        
        return ''.join(result)
    
    def _transform_homophone_symbols(self, text: str) -> str:
        """使用谐音字和符号替换"""
        # 常见替换映射
        replacements = {
            '死': '5', '杀': 'sa', '爱': '4', '草': 'c奥',
            '操': 'c奥', 'fuck': 'f**k', '傻逼': '傻b', '政府': 'zf',
            '国家': 'gj', '领导': 'ld', '敏感': 'm感', '封锁': 'f锁',
            '河蟹': '和谐', '自由': 'zy', '民主': 'mz', '独裁': 'dj',
            '屏蔽': 'pb', '和谐': 'hx', '真相': 'zx', '报道': 'bd',
            '抗议': 'ky', '示威': 'sw', '游行': 'yx', '集会': 'jh'
        }
        
        result = text
        for original, replacement in replacements.items():
            if original in result:
                result = result.replace(original, replacement)
        
        return result
    
    def _transform_multilingual_mix(self, text: str) -> str:
        """多语言混合"""
        # 简单的中英文关键词替换
        cn_to_en = {
            '政府': 'government', '国家': 'country', '领导': 'leader',
            '批评': 'criticize', '抗议': 'protest', '言论': 'speech',
            '自由': 'freedom', '民主': 'democracy', '权利': 'rights',
            '问题': 'problem', '争议': 'controversy', '冲突': 'conflict',
            '经济': 'economy', '政策': 'policy', '危机': 'crisis',
            '谎言': 'lie', '真相': 'truth', '腐败': 'corruption',
            '审查': 'censorship', '监控': 'surveillance', 'propaganda': '宣传'
        }
        
        result = text
        for cn, en in cn_to_en.items():
            if cn in result and random.random() < 0.7:  # 70%几率替换
                result = result.replace(cn, en)
        
        return result
    
    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """保存生成结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存完整JSON
        json_path = os.path.join(self.config.output_dir, f"generated_texts_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 转换为DataFrame并保存CSV（只包含原始文本和基本信息）
        df_data = []
        for item in results:
            row = {
                "template_id": item["template_id"],
                "template_description": item["template_description"],
                "topic": item["topic"],
                "original_text": item["original_text"]
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(self.config.output_dir, f"generated_texts_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"已保存生成结果到 {json_path} 和 {csv_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="舆情对抗文本生成工具")
    parser.add_argument("--model", type=str, default="llama3", help="要使用的Ollama模型名称")
    parser.add_argument("--templates", type=str, default="prompt_templates.json", help="模板JSON文件路径")
    parser.add_argument("--topics", type=str, default="topics.json", help="话题JSON文件路径")
    parser.add_argument("--output", type=str, default="generated_texts", help="输出目录")
    parser.add_argument("--samples", type=int, default=20, help="要生成的样本总数")
    parser.add_argument("--temp", type=float, default=0.7, help="温度参数")
    parser.add_argument("--no-adversarial", action="store_true", help="不生成对抗样本")
    
    args = parser.parse_args()
    
    # 创建示例话题文件（如果不存在）
    if not os.path.exists(args.topics):
        example_topics = [
            {
                "topic": "房价调控政策",
                "event": "最新房地产税收政策出台",
                "group": "年轻购房者"
            },
            {
                "topic": "教育改革",
                "event": "义务教育阶段减负新政策",
                "group": "学生家长"
            },
            {
                "topic": "医疗保障",
                "event": "新冠疫情后医疗保险调整",
                "group": "老年人"
            },
            {
                "topic": "环境保护",
                "event": "限塑令升级",
                "group": "环保人士"
            },
            {
                "topic": "网络安全",
                "event": "某科技巨头数据泄露事件",
                "group": "普通用户"
            }
        ]
        with open(args.topics, 'w', encoding='utf-8') as f:
            json.dump(example_topics, f, ensure_ascii=False, indent=2)
        logger.info(f"已创建示例话题文件: {args.topics}")
    
    # 配置生成器
    config = GenerationConfig(
        model=args.model,
        temperature=args.temp,
        templates_path=args.templates,
        topics_path=args.topics,
        output_dir=args.output,
        apply_adversarial=not args.no_adversarial
    )
    
    # 创建生成器并运行
    generator = TextGenerator(config)
    results = generator.generate_texts(num_samples=args.samples)
    
    logger.info(f"成功生成 {len(results)} 条文本样本")

if __name__ == "__main__":
    main()