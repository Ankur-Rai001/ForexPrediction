
# authdict = {"SECRET_KEY":"6no!r0#)gf(y%$9fb#u*9_!t=!8v5_sorv8k^nzs!5gwcxj#v!"}
    
from dotenv import load_dotenv
import os

load_dotenv()

# Get the SECRET_KEY
# secret_key = os.getenv('SECRET_KEY')
# print("secret_key =====>",secret_key)
authdict = {
    "SECRET_KEY": os.getenv("SECRET_KEY")
}