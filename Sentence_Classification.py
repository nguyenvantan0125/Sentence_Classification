    # Import module was created
import module as tx
    # Read data from CSV - đọc data từ file csv
df = tx.readcsv("texttrain.csv")
df.rename(columns={0:'feature',1:'label'},inplace=True)

    # create corpus - tạo bộ văn bản
corpus = df['feature'].values.tolist()

    # create label - tạo label 
y = df['label'].values.tolist()
    
    # Entry sentence relate to greeting or asking weather(Vietnamese requirement)
    # Nhập nhập những câu về chào hỏi hoặc hỏi về thời tiết (yêu cầu nhập tiếng việt)
print("\nEntry sentence relate to greeting OR asking weather(Vietnamese requirement)")
print("\nNhập nhập những câu về chào hỏi HOẶC hỏi về thời tiết (yêu cầu nhập tiếng việt)")
test_data = [str(input('entry: '))]

    # Model SVC
model_SVC = tx.SVC_linear(corpus,y,test_data)
model_SVC.processing()

    # Model Navie Bayes 
model_NB = tx.NavieBayes(corpus,y,test_data)
model_NB.processing()

    # Compute Cosine Similarity
con_sim = tx.Cosine_Sim(corpus,y,test_data)
con_sim.processing()
