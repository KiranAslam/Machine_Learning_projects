from textblob import TextBlob

text1 = "I am so happy and excited about this project!"
text2 = "This error is very annoying and frustrating."
blob1=TextBlob(text1)
blob2=TextBlob(text2)

print(f"Text 1 sentiment: {blob1.sentiment.polarity}")
print(f"Text 2 sentiment: {blob2.sentiment.polarity}")