import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import pickle

# Load and preprocess the data
df = pd.read_csv("train.csv")
df = df.fillna('')
df = df.drop(['id', 'title', 'author'], axis=1)

port_stem = PorterStemmer()

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

df['text'] = df['text'].apply(stemming)

# Prepare the data for LSTM model
x = df['text'].values
y = df['label'].values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Define the LSTM model----------------------------------------------------------------------------
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=2)

# Save the tokenizer and model
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
model.save('model.h5')

# Load the tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = tf.keras.models.load_model('model.h5')

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    input_data = tokenizer.texts_to_sequences(input_data)
    input_data = pad_sequences(input_data, maxlen=100)
    prediction = model.predict(input_data)
    return int(prediction.round())

# Test the function
val = fake_news("""In these trying times, Jackie Mason is the Voice of Reason. [In this week’s exclusive clip for Breitbart News, Jackie discusses the looming threat of North Korea, and explains how President Donald Trump could win the support of the Hollywood left if the U. S. needs to strike first.  “If he decides to bomb them, the whole country will be behind him, because everybody will realize he had no choice and that was the only thing to do,” Jackie says. “Except the Hollywood left. They’ll get nauseous. ” “[Trump] could win the left over, they’ll fall in love with him in a minute. If he bombed them for a better reason,” Jackie explains. “Like if they have no transgender toilets. ” Jackie also says it’s no surprise that Hollywood celebrities didn’t support Trump’s strike on a Syrian airfield this month. “They were infuriated,” he says. “Because it might only save lives. That doesn’t mean anything to them. If it only saved the environment, or climate change! They’d be the happiest people in the world. ” Still, Jackie says he’s got nothing against Hollywood celebs. They’ve got a tough life in this country. Watch Jackie’s latest clip above.   Follow Daniel Nussbaum on Twitter: @dznussbaum """)
if val == 0:
    print('reliable')
else:
    print('unreliable')
