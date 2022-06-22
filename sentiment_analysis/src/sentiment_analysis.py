import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str, default = "IMDB Dataset.csv",
                        help='please input a csv file')
    
    args = parser.parse_args()
    input_data = args.input_data_file
    df_review = pd.read_csv(input_data)
    df_positive = df_review[df_review['sentiment']=='positive'][:9000]
    df_negative = df_review[df_review['sentiment']=='negative'][:1000]
    df_review_imb = pd.concat([df_positive, df_negative])
    length_negative = len(df_review_imb[df_review_imb['sentiment']=='negative'])
    df_review_positive = df_review_imb[df_review_imb['sentiment']=='positive'].sample(n=length_negative)
    df_review_non_positive = df_review_imb[~(df_review_imb['sentiment']=='positive')]

    df_review_bal = pd.concat([
        df_review_positive, df_review_non_positive
    ])
    df_review_bal.reset_index(drop=True, inplace=True)
    train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)
    train_x, train_y = train['review'], train['sentiment']
    test_x, test_y = test['review'], test['sentiment']
    tfidf = TfidfVectorizer(stop_words='english')
    train_x_vector = tfidf.fit_transform(train_x)
    test_x_vector = tfidf.transform(test_x)
    svc = SVC(kernel='linear')
    svc.fit(train_x_vector, train_y)
    print(classification_report(test_y, 
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative']))
    
    #set the parameters
    parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
    svc = SVC()
    svc_grid = GridSearchCV(svc,parameters, cv=5)

    svc_grid.fit(train_x_vector, train_y)
    print(svc_grid.best_params_)
    print(svc_grid.best_estimator_)