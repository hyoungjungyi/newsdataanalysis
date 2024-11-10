import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# CNN 및 NYTimes 기사 링크 수집 함수
def get_cnn_article_links():
    url = 'https://www.cnn.com/'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access {url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []

    # /202로 시작하는 CNN 뉴스 링크 찾기
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/202'):
            full_url = 'https://www.cnn.com' + href
            article_links.append(full_url)
            if len(article_links) >= 50:
                break

    return list(set(article_links))  # 중복 제거 후 반환

def get_nytimes_article_links():
    url = 'https://www.nytimes.com/'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access {url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []

    # /202로 시작하는 NYTimes 뉴스 링크 찾기
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/202' in href:
            full_url = 'https://www.nytimes.com' + href if href.startswith('/') else href
            article_links.append(full_url)
            if len(article_links) >= 50:
                break

    return list(set(article_links))  # 중복 제거 후 반환

# 기사 본문 수집 함수
def fetch_article_content(urls):
    articles = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = [p.get_text() for p in soup.find_all('p')]
                articles.append(' '.join(paragraphs))
            else:
                print(f"Failed to retrieve article from {url}")
        except requests.exceptions.RequestException:
            print(f"Timeout or connection error for {url}")
    return articles

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 모델 정확도 출력 함수
def print_accuracy(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# 기사 수집 및 전처리
cnn_links = get_cnn_article_links()
nyt_links = get_nytimes_article_links()
article_urls = list(set(cnn_links + nyt_links))
raw_articles = fetch_article_content(article_urls)
preprocessed_articles = [preprocess_text(article) for article in raw_articles if article]

# 특성 추출 (CountVectorizer와 문서 길이 추가)
vectorizer = CountVectorizer(max_features=1000)
X_count = vectorizer.fit_transform(preprocessed_articles).toarray()
doc_lengths = np.array([len(text.split()) for text in preprocessed_articles]).reshape(-1, 1)
X = np.hstack((X_count, doc_lengths))

# 라벨 예시 (절반은 0, 나머지 절반은 1)
y = [0] * (len(preprocessed_articles) // 2) + [1] * (len(preprocessed_articles) - len(preprocessed_articles) // 2)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 정의 및 하이퍼파라미터 튜닝
models = {
    "Naive Bayes": MultinomialNB(),  # 스케일링 없이 원본 데이터 사용
    "Logistic Regression": GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}, cv=3),
    "Decision Tree": GridSearchCV(DecisionTreeClassifier(), {'max_depth': [5, 10, 20]}, cv=3),
    "SVM": GridSearchCV(SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, cv=3),
    "Multilayer Perceptron": GridSearchCV(MLPClassifier(max_iter=1000), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}, cv=3)
}

# 모델 학습 및 결과 출력
for name, model in models.items():
    if name == "Naive Bayes":
        print_accuracy(name, model, X_train, y_train, X_test, y_test)
    else:
        print_accuracy(name, model, X_train, y_train, X_test, y_test)
