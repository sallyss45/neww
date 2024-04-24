from flask import Flask, render_template, request
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            subject = request.form.get('subject')

            PATH = "C:\\web drivers\\chromedriver-win32\\chromedriver.exe"
            service = Service(PATH)
            driver = webdriver.Chrome(service=service)
            driver.get("https://twitter.com/login")
            sleep(3)
            username_field = driver.find_element(By.XPATH, "//input[@name='text']")
            username_field.send_keys(username)

            next_button = driver.find_element(By.XPATH, "//span[contains(text(),'Next')]")
            next_button.click()

            sleep(3)
            password_field = driver.find_element(By.XPATH, "//input[@name='password']")
            password_field.send_keys(password)

            log_in = driver.find_element(By.XPATH, "//span[contains(text(),'Log in')]")
            log_in.click()

            sleep(5)

            search_box = driver.find_element(By.XPATH, "//input[@data-testid='SearchBox_Search_Input']")
            search_box.send_keys(subject)
            search_box.send_keys(Keys.ENTER)

            sleep(3)
            Latest = driver.find_element(By.XPATH, "//span[contains(text(),'Latest')]")
            Latest.click()

            sleep(3)

            UserTags = []
            TimeStamps = []
            Tweets = []

            uniqueTweets = set()  # Using a set to store unique tweets

            while len(uniqueTweets) < 100:
                articles = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")

                for article in articles:
                    try:
                        UserTag = article.find_element(By.XPATH, ".//div[@data-testid='User-Name']").text
                        UserTags.append(UserTag)

                        TimeStamp = article.find_element(By.XPATH, ".//time").get_attribute('datetime')
                        TimeStamps.append(TimeStamp)

                        Tweet = article.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text

                        # Check if the tweet is unique before adding
                        if Tweet not in uniqueTweets:
                            Tweets.append(Tweet)
                            uniqueTweets.add(Tweet)

                    except Exception as e:
                        print(f"Exception occurred while scraping tweets: {e}")
                        # You can raise the exception if needed for debugging
                        # raise

                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                sleep(5)

            # Ensure that the lists are trimmed to only include the desired number of tweets
            UserTags = UserTags[:100]
            TimeStamps = TimeStamps[:100]
            Tweets = Tweets[:100]

            driver.quit()
            df = pd.DataFrame({'UserTag': UserTags, 'TimeStamp': TimeStamps, 'Tweets': Tweets})

            # Define the sentiment analysis model and tokenizer
            roberta = "cardiffnlp/twitter-roberta-base-sentiment"
            model = AutoModelForSequenceClassification.from_pretrained(roberta)
            tokenizer = AutoTokenizer.from_pretrained(roberta)

            labels = ['Negative', 'Neutral', 'Positive']
            # Define lists to store the results
            negative_scores = []
            neutral_scores = []
            positive_scores = []

            for tweet in df['Tweets']:
                # Convert tweet to string if it's not already
                if not isinstance(tweet, str):
                    tweet = str(tweet)

                # Encode the tweet
                encoded_tweet = tokenizer(tweet, return_tensors='pt')

                # Get the model output
                output = model(**encoded_tweet)

                # Get the scores and apply softmax
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)

                # Store scores in separate lists
                negative_scores.append(scores[0])
                neutral_scores.append(scores[1])
                positive_scores.append(scores[2])

            # Add the sentiment scores to the DataFrame
            df['Negative_Score'] = negative_scores
            df['Neutral_Score'] = neutral_scores
            df['Positive_Score'] = positive_scores

            # Save DataFrame to CSV
            df.to_csv('tweet.csv', index=False)

            positive = df['Positive_Score'].sum()
            negative = df['Negative_Score'].sum()
            neutral = df['Neutral_Score'].sum()

            return render_template('display.html', df=df)

        except Exception as e:
            print(f"Exception occurred in the scraping process: {e}")
            return render_template('error.html', error_message="An error occurred during scraping.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
