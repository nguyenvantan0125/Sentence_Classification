    # Import module was created
import textclassification as tx

    # Read data from CSV - đọc data từ file csv
df = tx.readcsv("texttrain.csv")
df.rename(columns={0:'feature',1:'label'},inplace=True)

    # create corpus - tạo bộ văn bản
corpus = df['feature'].values.tolist()

    # create label - tạo label 
y = df['label'].values.tolist()
    
    # Entry sentence relate to greeting or asking weather(require Vietnamese)
    # Nhập nhập những câu về chào hỏi hoặc hỏi về thời tiết (yêu cầu nhập tiếng việt)
print("Entry sentence relate to greeting OR asking weather(require Vietnamese)")
print("Nhập nhập những câu về chào hỏi HOẶC hỏi về thời tiết (yêu cầu nhập tiếng việt)")
test_data = [str(input('entry: '))]

    # tokenizer Vietnamese - thực hiện tách từ 
corpus = tx.vitoken(corpus)
test_data = tx.vitoken(test_data)

    # onehotvector - Mã hóa bộ văn bản
vect = tx.vectorizer.countvector(corpus)
bow = tx.vectorizer.bag_of_words(vect,corpus)

    # create training set 
X_train,X_test= tx.preprocessing.processing(vect,corpus,test_data)


# SVC_linear training AND show result
tx.models_selections.model_SVC_linear (X_train, y , X_test)


# SVC_rbf training AND show result
tx.models_selections.model_SVC_rbf(X_train, y , X_test)


# Naive Bayers AND show result
tx.models_selections.pipeline_NB(corpus,y,test_data)












