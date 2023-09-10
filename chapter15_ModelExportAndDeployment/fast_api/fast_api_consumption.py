import requests
from PIL import Image
from io import BytesIO

# URL for the FastAPI application
url = "http://localhost:8000/segment/"

# Image file to segment
filename = "food_image.jpg"

# Open the image in binary mode
with open(filename, "rb") as file:
    # Post the image to the API
    response = requests.post(url, files={"file": file})

# Check if the request was successful
if response.status_code == 200:
    # Open the image from the response
    image = Image.open(BytesIO(response.content))

    # Display the image
    image.show()
else:
    print(f"Request failed with status code {response.status_code}")
