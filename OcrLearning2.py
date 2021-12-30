from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd= R'C:\Program Files\Tesseract-OCR\tesseract'
str = pytesseract.image_to_string(Image.open('D:\OcrTest\Music.png'), lang='kor')



print(str)
