import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def set_connection():
    # Fetch variables
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    # Connect to the database
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        print("Connection successful!")
        
        # Create a cursor to execute SQL queries
        cur = connection.cursor()
        return cur
    
    except Exception as e:
        print(f"Failed to connect: {e}")

set_connection()

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Buat canvas PDF
pdf_file = "contoh_sederhana.pdf"
c = canvas.Canvas(pdf_file, pagesize=A4)

# Judul
c.setFont("Helvetica-Bold", 16)
c.drawString(100, 800, "Laporan Otomatis - Contoh Sederhana")

# Paragraf teks
c.setFont("Helvetica", 12)
text = (
    "Ini adalah contoh sederhana pembuatan file PDF menggunakan ReportLab.\n"
    "ReportLab memungkinkan kita untuk membuat dokumen PDF secara dinamis\n"
    "dengan kontrol penuh atas teks, layout, dan elemen grafis lainnya."
)
text_object = c.beginText(100, 770)
for line in text.split("\n"):
    text_object.textLine(line)

c.drawText(text_object)

# Simpan dan tutup file
c.showPage()
c.save()

print(f"PDF '{pdf_file}' berhasil dibuat.")

