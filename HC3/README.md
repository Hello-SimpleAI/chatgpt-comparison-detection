**Updates:**

** 2023/01/18, the HC3 datasets (English & Chinese) are available at 🤗 HuggingFace & ModelScope!

** 2023/01/18, HC3 数据集 (中文版 & 英文版) 已经上线 🤗 HuggingFace & ModelScope!


---

**HC3 (Engllish) @HuggingFace :**\
https://huggingface.co/datasets/Hello-SimpleAI/HC3

**HC3 (Chinese) @HuggingFace :**\
https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese

**HC3 (英文版) @ModelScope :**\
https://www.modelscope.cn/datasets/simpleai/HC3

**HC3 (中文版) @ModelScope :**\
https://www.modelscope.cn/datasets/simpleai/HC3-Chinese


<img width="512" alt="image" src="https://user-images.githubusercontent.com/37113676/213214747-f03e15e6-6601-49ff-8c48-ace14f114572.png">

---
We release the **training and testing corpus** used in our ChatGPT detector(s), including:
- [不过滤-全文](https://drive.google.com/drive/folders/1st2HLKSOVajekT_RvYRP2sFAfnLGC2g-?usp=sharing) contains question-answer pairs, where the answer contains **all** the text **without** filtering indicating words. Both English and Chinese verions are provided.
- [不过滤-句子](https://drive.google.com/drive/folders/1m5X5tiiBIJiFGGhDviAyk1ipz5WxftPs?usp=sharing) contains question-answer pairs, where the answer contains only one **splited** sentence **without** filtering indicating words. Both English and Chinese verions are provided.
- [过滤-全文](https://drive.google.com/drive/folders/1qZF6NaTdaHrMvqljpnVGl6gIHgaCT49i?usp=sharing) contains question-answer pairs, where the answer contains **all** the text and **filters** the indicating words. Both English and Chinese verions are provided.
- [过滤-句子](https://drive.google.com/drive/folders/1uY1-Ef8S_oxRFDofQSf649V_ESzr7hiU?usp=sharing) contains question-answer pairs, where the answer contains only one **splited** sentence and **filters** the indicating words. Both English and Chinese verions are provided.

---
Besides, we provide [DEMO](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/HC3/demo_indicating_words.ipynb) for removing the indicating words in Human or ChatGPT answers, including both English and Chinese corpus.
