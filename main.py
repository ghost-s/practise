import requests

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
}  # Modi...前面不能有空格
url = 'https: //image.baidu.com/search/acjson?'
params = {
    'tn': 'resultjson_com',
    'logid': '9134715202025715844',
    'ipn': 'rj',
    'ct':'201326592',
    'is': '',
    'fp':'result',
    'fr': '',
    'word': '赵丽颖剧照',
    'queryWord': '赵丽颖剧照',
    'cl': '2',
    'lm': '-1',
    'ie': 'utf-8',
    'oe': 'utf-8',
    'adpicid': '',
    'st': '-1',
    'z': '',
    'ic': '',
    'hd': '',
    'latest': '',
    'copyright': '',
    's': '',
    'se': '',
    'tab': '',
    'width': '',
    'height': '',
    'face': '0',
    'istype': '2',
    'qc': '',
    'nc': '1',
    'expermode': '',
    'nojc': '',
    'isAsync': '',
    'pn': '60',
    'rn': '50',  # 改变rn值可以改变爬取的图片的数量
    'gsm': '3c',
}

result = requests.get(url=url, headers=header, params=params)
print(result.text)

result = result.json()
data = result['data']
del data[-1]
key_list = []
for info in 'data':
    key_list.append(info['thumbURL'])
n = 1
for img_url in 'key_list':
    img_data = requests.get(url=img_url, headers=header).content
    img_path = 'D:/zly/' + str(n) + '.jpg'
    with open(img_path, 'wb') as fp:
        fp.write(img_data)
        print(n)
        n += 1
