from icrawler.builtin import BingImageCrawler

# Tạo crawler tải ảnh dâu bị hư
keywords = [
    'spoiled strawberries',
    'moldy strawberries', 
    'decaying strawberries',
    'overripe strawberries',
    'bad strawberries',
    'strawberry grey mold',
    'strawberry anthracnose',
    'strawberry powdery mildew',
    'strawberry leather rot',
    'strawberry rhizopus rot',
    'rotten strawberries in container',
    'rotten strawberry on plant',
    'rotten strawberries on plant',
    'anthracnose strawberry', 
]

for kw in keywords:
    # Tạo thư mục con cho từng từ khóa để dễ quản lý
    sub_dir = f'dataset/{kw.replace(" ", "_")}'
    crawler = BingImageCrawler(storage={'root_dir': sub_dir})
    crawler.crawl(keyword=kw, max_num=100)  