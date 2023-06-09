"""
Query ChatGPT via Selenium.

@Time : 2022/12/17 15:00
@Author : izhx (Xin Zhang)
@File : query.py
@Project : ChatGPT

pip install undetected-chromedriver webdriver-manager

"""

import argparse
import glob
import json
import os
import random
import re
import signal
import time
from typing import Dict, List

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.remote.webelement import WebElement
import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager

Anwser = List[str]


def printf(*args):
    print(time.asctime(), "-", *args)


EMOJI_REGEX = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "\\*\u20e3"
    "#\u20e3"
    "]+",
    flags=re.UNICODE,
)
# COLON_REGEX = re.compile(r"[:\s]{4,}")


def remove_emoji(text):
    text = EMOJI_REGEX.sub(r"", text)
    # text = COLON_REGEX.sub(r"", text)
    return text.strip()


class NeedSleep(Exception):
    pass


class NeedRefresh(Exception):
    pass


class NeedNewChat(Exception):
    pass


class WillBeSkipped(Exception):
    pass


class NoRegenerationButton(Exception):
    pass


class Selector:
    input_textarea = 'textarea.resize-none'
    regenerate_button = 'button.btn.flex.justify-center.gap-2'
    new_chat_link = 'nav>a.border'
    content_row = '.w-full.border-b.text-gray-800.group'
    chargpt_row = '.w-full.border-b.text-gray-800.group.bg-gray-50'


def input_textarea() -> WebElement:
    textarea = driver.find_element(By.CSS_SELECTOR, Selector.input_textarea)
    tabindex = textarea.get_attribute('tabindex')
    rows = textarea.get_attribute('rows')
    assert tabindex == '0' and rows == '1'
    return textarea


def regenerate_response_button() -> WebElement:
    buttons = driver.find_elements(By.CSS_SELECTOR, Selector.regenerate_button)
    for b in buttons:
        if "regenerate response" in b.text.lower():
            return b
    raise NoRegenerationButton


def chat_chatable(*args):
    # 检查是否进入了聊天界面，通过寻找输入框 (textarea) 实现
    driver.implicitly_wait(random.uniform(0.5, 1))
    try:
        input_textarea()
    except NoSuchElementException:
        # printf("当前不在聊天界面中!")
        return False
    return True


def verify_human():
    try:
        inputs = driver.find_elements(By.TAG_NAME, 'input')
        for i in inputs:
            if 'you are human' in i.get_attribute('value'):
                i.click()
                printf("Auto click human verify.")
                time.sleep(3)
                return
    except NoSuchElementException:
        pass
    for selector in ('#spinner-icon', 'span.mark'):
        try:
            cf_box = driver.find_element(By.CSS_SELECTOR, selector)
            printf("Auto click cloudflare.")
            cf_box.click()
            time.sleep(3)
            return
        except NoSuchElementException:
            pass


def refresh(max_trial: int = 5):
    driver.get("https://chat.openai.com/")
    for _ in range(max_trial):
        time.sleep(3)
        verify_human()
        if chat_chatable():
            break
    else:
        raise NeedSleep


def new_chat(max_trial: int = 5):
    try:
        a = driver.find_element(By.CSS_SELECTOR, Selector.new_chat_link)
        assert "new chat" in a.text.lower()  # 服了，这大小写有啥可改的
        a.click()
    except NoSuchElementException:
        refresh()
        return

    for _ in range(max_trial):
        time.sleep(1)
        if chat_chatable():
            break
    else:
        refresh()


def close_driver(_signal, frame):
    if driver is not None:
        driver.quit()


def login(max_trial: int = 5) -> bool:
    # TODO check login page and wait
    driver.get("https://chat.openai.com/")

    success = False
    for _ in range(max_trial):
        time.sleep(3)
        just = 0
        while driver.title == 'Just a moment...':
            verify_human()
            just += 1
            if just >= max_trial:
                driver.get("https://chat.openai.com/")
                just = 0
            time.sleep(3)
        if driver.title == '':
            divs = driver.find_elements(By.TAG_NAME, 'div')
            text = ' '.join(d.text for d in divs)
            if (
                'Log in with your OpenAI account to continue' in text
                or 'Welcome back' in text or 'Enter your password' in text
                or 'Create your account' in text
            ):
                printf("未自动登录，请手动登陆并进入聊天界面，然后按任意键继续")
                input()
            time.sleep(2)
        success = chat_chatable()
        if success:
            break
    else:
        printf(f"超过最大尝试次数 (max_trial = {max_trial}), 请重新启动脚本.")
        exit()

    printf("登陆成功!")
    return True


def send_query(query: str, max_trial: int = 5):
    """
    发送问题
    """
    query = remove_emoji(query)

    def send():
        rows = driver.find_elements(By.CSS_SELECTOR, Selector.content_row)

        textarea = input_textarea()
        textarea.send_keys(query)
        if textarea.text != query:
            return False

        time.sleep(0.5)
        textarea.find_element(By.XPATH, './following-sibling::button[1]').click()
        time.sleep(0.5)

        new_rows = driver.find_elements(By.CSS_SELECTOR, Selector.content_row)
        if len(new_rows) >= len(rows):
            return True
        raise RuntimeError

    for _ in range(max_trial):
        if send():
            break
        refresh()
    else:
        printf("发送失败")
        raise NeedSleep("发送失败")

    started = False
    while not started:
        # Wait for chatgpt start talking.
        rows = driver.find_elements(By.CSS_SELECTOR, Selector.content_row)
        # and last_query in rows[-2].text 先不判断这个了，都是单轮
        # TODO 偶发 rows[-2].text 是 last_query 的截断，还没找到原因
        if len(rows) > 1 and len(rows) % 2 == 0:
            if 'bg-gray-50' in rows[-1].get_attribute('class'):
                started = True
        time.sleep(5)


other_tags = set()

def parse_chat_row(response: WebElement) -> Anwser:
    """
    解析 ChatGPT 返回内容, 并: 1)保持列表信息.
    """
    children = response.find_elements(By.XPATH, './*')
    anwser = list()
    for tag in children:
        if tag.tag_name == 'ol':
            rows = tag.find_elements(By.XPATH, './*')
            for i, r in enumerate(rows, 1):
                anwser.append(f'{i}. {r.text}')
        else:
            if tag.tag_name not in other_tags:
                other_tags.add(tag.tag_name)
                printf('new tag:', tag.tag_name)
            anwser.append(tag.text)
    return anwser


def get_response() -> Anwser:
    chat_row = driver.find_elements(By.CSS_SELECTOR, Selector.chargpt_row)[-1]

    # Wait for chatgpt finish.
    feedback = chat_row.find_elements(By.CSS_SELECTOR, 'button')[-1]
    while not feedback.is_displayed():
        time.sleep(3)

    content = chat_row.find_element(By.CSS_SELECTOR, '.flex.flex-col.gap-3 div div')
    response = content.text
    if 'text-red' in content.get_attribute('class'):
        # error
        printf("OpenAI Error:", response)
        if 'Too many requests' in response:
            # 'Too many requests in 1 hour.'
            # 'Too many requests, please slow down'
            raise NeedSleep
        elif 'something seems to have gone wrong' in response:
            raise NeedSleep
        elif 'Internal server error' in response:
            raise NeedNewChat
        elif 'An error occurred.' in response:
            raise NeedRefresh
        elif 'Only one message at a time.' in response:
            raise NeedRefresh
        elif 'an error while processing your request' in response:
            raise NeedNewChat
        elif 'can retry your request' in response:
            raise NeedNewChat
        elif 'network error' in response:
            raise WillBeSkipped
        else:
            raise RuntimeError("New error!")

    return parse_chat_row(content)


def chatgpt(query: str, num_answers: int = 1) -> List[Anwser]:
    """
    Query ChatGPT by the given `query`, generate `num_answers` times and return.
    """
    printf("Query:", query)

    if not chat_chatable():
        refresh()

    send_query(query)

    answers = list()
    while True:
        time.sleep(2)
        try:
            response = get_response()
        except NeedNewChat:
            new_chat()
            send_query(query)
            continue
        except NeedRefresh:
            refresh()
            send_query(query)
            continue

        printf("Got:", response[0][:36], '...')
        answers.append(response)

        text = ''.join(response)
        if '我无法理解您的问题' in text:
            break
        factor = 0
        for phrase in (
            "I'm sorry", "provide more", "很抱歉", "无法回答", "我是一个", "语言模型",
            "计算机程序", "无法确定"
        ):
            if phrase in text:
                factor += 1
        if factor >= 2:
            break

        if len(answers) >= num_answers:
            break

        time.sleep(random.uniform(3, 5))
        try:
            regenerate_response_button().click()
        except NoRegenerationButton:
            new_chat()
            send_query(query)

    return answers


def process_data(
    questions: Dict[int, Dict], prefix: str = 'data/', num_answers: int = 1
) -> List[List[str]]:
    time.sleep(1)
    for i, q in questions.items():
        timer = time.time()
        try:
            answers = chatgpt(q['query'], num_answers)
        except WillBeSkipped:
            time.sleep(random.uniform(1, 5))
            new_chat()
            continue
        q['chats'] = answers
        with open(f'{prefix}.{i}.json', mode='w', encoding='utf8') as file:
            # q.pop('query')
            json.dump(q, file, indent=2, ensure_ascii=False)
            printf(f'Saved at', file.name)
        time.sleep(random.uniform(1, 5))
        new_chat()
        timer = time.time() - timer
        # if timer < 45:
        #     time.sleep(45 - timer)


def read_data(filename: str) -> Dict[int, Dict]:
    if isinstance(filename, str) and os.path.exists(filename):
        printf("输入文件:", filename)
    else:
        printf("文件不存在:", filename)
        exit()

    raw = list()
    with open(filename, encoding='utf8') as file:
        for line in file:  # {'query': '....', **kwargs}
            obj = json.loads(line)
            # obj['query'] = "我有一个计算机相关的问题，请用中文回答，什么是 " + obj['title']
            raw.append(obj)

    data = {i: r for i, r in enumerate(raw)}
    if len(data):
        printf(f"成功读取 {len(data)} 句, 第一个: {data[0]}")
    else:
        printf("未读取到任何数据，请检查输入文件")
        exit()
    return data


def collect_result(input_path, output_dir, json_prefix):
    files = glob.glob(f'{json_prefix}.*.json')
    # 按原始顺序整合为jsonline
    output_path = os.path.join(output_dir, 'chat_' + os.path.basename(input_path))
    with open(output_path, mode='w', encoding='utf8') as writter:
        for p in files:
            with open(p, encoding='utf8') as reader:
                obj = json.load(reader)
            writter.write(json.dumps(obj, ensure_ascii=False))
            writter.write('\n')
        printf(f"Write {len(files)} lines to", writter.name)


def main(args: argparse.Namespace, json_prefix: str):

    login()

    data = read_data(args.raw)

    def refresh_data():
        # data = read_data(args.raw)
        files = glob.glob(f'{json_prefix}.*.json')
        for p in files:
            index = int(p.split('.')[-2])
            data.pop(index, None)
        if len(files):
            printf(f"已完成 {len(files)} 句.")
        return data

    iteraiton = 0
    while len(data):
        data = refresh_data()  # to recover history.
        print('')
        printf(f"第 {iteraiton} 次尝试，目前剩余 {len(data)} 句.")
        try:
            process_data(data, json_prefix, args.num_answers)
        except NeedSleep:
            printf("网络错误，睡眠一会")
            time.sleep(5 * 60)
        iteraiton += 1
    print('')

    collect_result(args.raw, args.output_dir, json_prefix)
    return


if __name__ == "__main__":
    _PARSER = argparse.ArgumentParser("")
    _PARSER.add_argument(
        '-r', '--raw', type=str, help="原始数据文件, jsonline",
        default='wild_data/Wiki_Concepts/baidu_baike_is_p3.json'
        # {'query': '....', **kwargs}
    )
    _PARSER.add_argument(
        '-o', '--output-dir', type=str, default='data', help="输出保存文件夹"
    )
    _PARSER.add_argument(
        '-c', '--collect', default=False, action="store_true", help="直接整理现有结果"
    )
    _PARSER.add_argument(
        '-n', '--num-answers', type=int, default=1, help="输出结果数量"
    )
    _PARSER.add_argument(
        '-s', '--sock5', type=str, default='13659', help="本地 sock5 代理端口"
    )
    _PARSER.add_argument(
        '-d', '--debugger', type=str, default=None, help="本地 chrome debugger 端口"
    )
    _PARSER.add_argument(
        '--user-data-dir', type=str, default='.chrome_data', help="本地 chrome profile 位置"
    )
    _PARSER.add_argument(
        '--driver-dir', type=str, default='.drivers', help="本地 chrome driver 位置"
    )
    _ARGS = _PARSER.parse_args()

    # 临时存储相关准备
    tmp_dir = os.path.join(_ARGS.output_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    _json_prefix = os.path.join(tmp_dir, os.path.basename(_ARGS.raw))

    if _ARGS.collect:
        collect_result(_ARGS.raw, _ARGS.output_dir, _json_prefix)
        exit()

    # set proxy
    _OPTIONS = uc.ChromeOptions()
    _OPTIONS.add_argument('--proxy-server=socks5://127.0.0.1:' + _ARGS.sock5)
    # _OPTIONS.add_argument('--no-sandbox')
    if _ARGS.debugger is not None:
        _OPTIONS.add_experimental_option('debuggerAddress', '127.0.0.1:' + _ARGS.debugger)

    _SERVICE = ChromeService(ChromeDriverManager(path=_ARGS.driver_dir).install())

    driver: uc.Chrome = None
    # init chrome driver
    _ARGS.user_data_dir = os.path.abspath(_ARGS.user_data_dir)
    try:
        driver = uc.Chrome(
            options=_OPTIONS,
            user_data_dir=_ARGS.user_data_dir,
            service=_SERVICE
        )
    except Exception as e:
        print(e)
        chrome_path = uc.find_chrome_executable().replace(' ', '\ ')
        print("如果显示连接 Chrome 超时, 请杀死所有 Chrome 进程, 然后尝试从命令行启动 Chrome :")
        print(f'{chrome_path} --remote-debugging-port=9222')
        print("或者更简单的办法是删除本地 selenuim 的 Chrome Profile :")
        print(f'rm -rf {_ARGS.user_data_dir}')
        exit()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, close_driver) # 定义捕获信号和关闭时的处理函数

    printf("pid", os.getpid(), ', args:', json.dumps(vars(_ARGS), indent=2))
    main(_ARGS, _json_prefix)
