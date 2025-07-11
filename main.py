import pandas as pd
from datetime import datetime, timedelta,date
import pandas as pd
import json
from decimal import Decimal

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np

import psycopg2
from dotenv import load_dotenv
import os

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from io import BytesIO
import requests

from pdf2image import convert_from_path
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

# Load environment variables from .env
load_dotenv()

def send_email(image_folder):
# CONFIG
    # image_folder = "pdf_image/5 - 9 May 2025"  # folder where images are stored
    allowed_extensions = {"jpg", "jpeg", "png"}
    sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
    from_email = "gerald@supertype.ai"
    to_email = ["geraldbryan9914@gmail.com","shusi.evelyn@gmail.com"]

    # Collect all image files from the folder
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.split(".")[-1].lower() in allowed_extensions
    ]

    # Create attachments list
    attachments = []
    for path in image_files:
        with open(path, "rb") as f:
            data = f.read()
            encoded_file = base64.b64encode(data).decode()

        attachment = Attachment(
            FileContent(encoded_file),
            FileName(os.path.basename(path)),
            FileType(f"image/{path.split('.')[-1]}"),
            Disposition("attachment")
        )
        attachments.append(attachment)

    # Create email
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject="All Images in Folder Attached",
        html_content = """
    <p>Hi,</p>
    <p>Please find all image attachments for IDX Weekly Highlights.</p>
    <p>Thank you<br><br><br>
    Best regards,<br>
    Gerald</p>
    """

    )

    # Add attachments
    message.attachment = attachments

    # Send email via SendGrid
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        print("Email sent successfully:", response.status_code)
    except Exception as e:
        print("Failed to send email:", str(e))

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

def fetch_query(query, cur):

    cur.execute(query)

    rows = cur.fetchall()

    colnames = [desc[0] for desc in cur.description]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=colnames)

    return df

def ffill_data(df,column):
    # Count how many NaNs are at the end
    reversed_vals = df[column][::-1]
    n_trailing_nans = reversed_vals.isna().cumprod().sum()

    # Forward fill everything except trailing NaNs
    if n_trailing_nans > 0:
        filled_part = df[column][:-n_trailing_nans].ffill()
        trailing_nans = pd.Series([np.nan] * n_trailing_nans, index=df.index[-n_trailing_nans:])
        df[f"{column}_ffill"] = pd.concat([filled_part, trailing_nans])
    else:
        df[f"{column}_ffill"] = df[column].ffill()

    return df

def draw_shrinking_text(pdf, text, max_width, x, y, font_name='Inter-Bold', initial_font_size=30, min_font_size=5, color=colors.white):
    """
    Draws text at (x, y) with shrinking font size if max_width is exceeded.

    Parameters:
    - pdf: ReportLab canvas object
    - text: The string to draw
    - max_width: Maximum allowed width for the text
    - x, y: Coordinates to draw the text
    - font_name: Font to use (default: 'Inter-Bold')
    - initial_font_size: Starting font size (default: 30)
    - min_font_size: Minimum font size allowed (default: 5)
    - color: Text color (default: white)
    """
    font_size = initial_font_size
    pdf.setFont(font_name, font_size)
    text_width = pdf.stringWidth(text, font_name, font_size)

    while text_width > max_width and font_size > min_font_size:
        font_size -= 1
        pdf.setFont(font_name, font_size)
        text_width = pdf.stringWidth(text, font_name, font_size)

    pdf.setFillColor(color)
    pdf.drawString(x, y, text)

def week_date():
    today = datetime.today()

    # Find Monday of this week
    start_of_week = today - timedelta(days=today.weekday())

    # Generate Monday to Friday dates (as date objects)
    weekdays = [(start_of_week + timedelta(days=i)).date() for i in range(5)]

    # Create DataFrame
    df = pd.DataFrame({"date": weekdays})

    return df

def full_week(db_data):
    df = week_date()
    
    df['date'] = df['date'].astype("str")
    db_data['date'] = db_data['date'].astype("str")

    df = df.merge(db_data, on="date", how='left')

    df['date'] = pd.to_datetime(df['date'])
    df['pct_change'] = (df['market_cap'].pct_change() * 100)
    df['pct_change'] = df['pct_change'].astype("float").round(2)

    df = ffill_data(df,"market_cap")

    return df

def date_generator(output):
    dt_start = week_date().iloc[0]['date']
    dt_end = week_date().iloc[-1]['date']

    if output == "cover":
        if dt_start.year == dt_end.year:
            if dt_start.month == dt_end.month:
                return f"{dt_start.day} - {dt_end.day} {dt_end.strftime('%b')} {dt_end.year}"
            else:
                return f"{dt_start.day} {dt_start.strftime('%b')} - {dt_end.day} {dt_end.strftime('%b')} {dt_end.year}"
        else:
            return f"{dt_start.day} {dt_start.strftime('%b')} {dt_start.year} - {dt_end.day} {dt_end.strftime('%b')} {dt_end.year}"
    if output == "calendar":
        if dt_start.year == dt_end.year:
            if dt_start.month == dt_end.month:
                return f"{dt_end.strftime('%B')} {dt_end.year}"
            else:
                return f"{dt_start.strftime('%B')} - {dt_end.strftime('%B')} {dt_end.year}"
        else:
            return f"{dt_start.strftime('%b')} {dt_start.year} - {dt_end.strftime('%B')} {dt_end.year}"
    
def custom_formatter(x, pos):
    """
    Formatter for large numbers, adding suffixes like B (billion), M (million), T (trillion), etc.
    """
    if x >= 1e12:  # Trillions
        return f'{x/1e12:.0f} T'
    elif x >= 1e9:   # Billions
        return f'{x/1e9:.0f} B'
    elif x >= 1e6:   # Millions
        return f'{x/1e6:.0f} M'
    else:
        return f'{x:.0f}'

# Plot    
def mcap_chart(hist_mcap):
    # Convert dates to numeric format
    date_nums = mdates.date2num(hist_mcap['date'])

    # Shift tick positions
    tick_spacing = (date_nums[-1] - date_nums[0]) / (len(date_nums) - 1)
    shift_factor = 0.05
    shifted_start = date_nums[0] + tick_spacing * shift_factor
    shifted_xs = [shifted_start + i * tick_spacing for i in range(len(date_nums))]

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Fonts
    inter_regular = fm.FontProperties(fname='asset/font/Inter-Regular.ttf')
    inter_semi_bold = fm.FontProperties(fname='asset/font/Inter-SemiBold.ttf')

    min_mcap = float(hist_mcap["market_cap"].min())
    max_mcap = float(hist_mcap["market_cap"].max())

    # Plot shifted line

    ax.plot(shifted_xs, hist_mcap["market_cap"],
            linewidth=8, marker='o', markersize=22,
            color="#F29942", markerfacecolor='#F29942', markeredgecolor='#F29942')

    ax.plot(shifted_xs, hist_mcap["market_cap"],
            linestyle=' ', marker='o', markersize=22,
            markerfacecolor='#F29942', markeredgecolor='#F29942')

    ax.plot(shifted_xs, hist_mcap["market_cap_ffill"],
            linewidth=8,
            color="#F29942",
            linestyle='dashed')

    # Market cap label
    for x, y in zip(shifted_xs, hist_mcap["market_cap"]):
        if np.isnan(float(y)):
            ax.text(x, ((hist_mcap["market_cap"].max() + hist_mcap["market_cap"].min())/2), "Exchange\nHoliday",
                fontsize=20, fontproperties=inter_semi_bold,
                ha='center', va='bottom', color="white")
        else:
            ax.text(x, float(y) * 1.05, f"IDR {format_number_short_2d(y)}",
                fontsize=20, fontproperties=inter_semi_bold,
                ha='center', va='bottom', color="#F29942")
            prev_y = float(y)

    # Percent change label
    for x, y, z in zip(shifted_xs, hist_mcap["market_cap"], hist_mcap["pct_change"]):
        if not np.isnan(z):
            label = f"{z:.2f}%"
            bbox = dict(
                boxstyle="round, pad=0.5",
                facecolor='#568475' if z >= 0 else '#D53E50',
                edgecolor='none',
                alpha=1
            )
            ax.text(x, float(y) * 1.17, label,
                    fontsize=20, fontproperties=inter_semi_bold,
                    ha='center', va='bottom', color="white", bbox=bbox)

    # Style ticks with spacing
    ax.tick_params(axis='x', labelsize=28, colors="white", pad=20)  # Add horizontal padding (below x-axis)
    ax.tick_params(axis='y', labelsize=28, colors="white", pad=15)  # Add vertical padding (left of y-axis)

    # Apply font to tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(inter_regular)
        label.set_fontsize(24)

    # Custom x-ticks
    ax.set_xticks(shifted_xs)
    ax.set_xticklabels(hist_mcap['date'].dt.strftime('%d %B'))

    # Fix xlim to respect shifted ticks
    padding = tick_spacing * 0.5  # Optional padding on the right
    ax.set_xlim(shifted_xs[0] - tick_spacing * 0.5, shifted_xs[-1] + padding)

    # Custom y-axis formatting
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

    # Grid and spines
    ax.yaxis.grid(True, alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('white')

    # Y-limits
    ax.set_ylim((min_mcap * 0.5), (max_mcap * 1.5))

    for i in range(0,4):
        ax.axvline(shifted_start+0.5+i, color='white', linestyle='dashed', linewidth=1, alpha=0.3)

    # Save and show
    plt.tight_layout()
    plt.savefig("asset/plot/my_plot.png", transparent=True, dpi=300)
    return plt

def ca_compilation(df_div,df_ipo,stock_split):
    df_week = week_date()

    df_div_proc = pd.DataFrame(df_div.groupby('ex_date')['symbol'].count()).reset_index()
    df_div_proc.columns = ['date','div']

    df_ipo_proc = pd.DataFrame(df_ipo.groupby('offering_end_date')['symbol'].count()).reset_index()
    df_ipo_proc.columns = ['date','ipo']

    df_stock_split_proc = pd.DataFrame(stock_split.groupby('date')['symbol'].count()).reset_index()
    df_stock_split_proc.columns = ['date','split']

    for i in [df_ipo_proc, df_div_proc, df_stock_split_proc]:
        df_week = df_week.merge(i, on='date', how="left")

    df_week = df_week.fillna(0)

    df_week['total'] = df_week['ipo'] + df_week['div'] + df_week['split']

    df_week[df_week.select_dtypes(include='number').columns] = df_week.select_dtypes(include='number').fillna(0).astype(int)

    return df_week

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def format_number_short_1d(num):
    """
    Convert number to abbreviated format:
    1,200,000 -> 1.2M
    1,500,000,000 -> 1.5B
    2,000,000,000,000 -> 2T
    """
    num = float(num)
    
    if abs(num) >= 1_000_000_000_000:  # Trillion
        return f"{num / 1_000_000_000_000:,.1f}T"
    elif abs(num) >= 1_000_000_000:  # Billion
        return f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:  # Million
        return f"{num / 1_000_000:.1f}M"
    else:
        return f"{num:,.0f}"
    
def format_number_short_2d(num):
    """
    Convert number to abbreviated format:
    1,200,000 -> 1.2M
    1,500,000,000 -> 1.5B
    2,000,000,000,000 -> 2T
    """
    num = float(num)
    
    if abs(num) >= 1_000_000_000_000:  # Trillion
        return f"{num / 1_000_000_000_000:,.2f}T"
    elif abs(num) >= 1_000_000_000:  # Billion
        return f"{num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:  # Million
        return f"{num / 1_000_000:.2f}M"
    else:
        return f"{num:,.0f}"
    
def create_weekly_report(hist_mcap, mcap_changes,top_gainers_losers,indices_changes,sectors_changes,top_3_comp_sectors,top_volume,top_value,df_ipo,stock_split,df_div,ca_comp):
    # Register Inter font
    pdfmetrics.registerFont(TTFont('Inter', 'asset/font/Inter-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('Inter-Bold', 'asset/font/Inter-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('Inter-Semi-Bold', 'asset/font/Inter-SemiBold.ttf'))

    # Setup Canvas
    width, height = 1200, 1500
    pdf = canvas.Canvas(f"pdf_output/idx_highlights - {date_generator('cover')}.pdf", pagesize=(width, height))

    # Cover
    num_week = round(datetime.today().day/7)
    if num_week <= 4 and num_week>0:
        pdf.drawImage(f'asset/page/cover-{num_week}.png', 0, 0, width, height)
    if num_week == 0:
        pdf.drawImage('asset/page/cover-1.png', 0, 0, width, height)
    else:
        pdf.drawImage('asset/page/cover-1.png', 0, 0, width, height)
    pdf.setFillColor(colors.white)
    pdf.setFont("Inter", 45)
    pdf.drawString(149, 765,  date_generator("cover"))

    plot = mcap_chart(hist_mcap)

    # Page 1
    pdf.showPage()
    pdf.drawImage('asset/page/page_1.png', 0, 0, width, height)

    # Chart
    pdf.drawImage("asset/plot/my_plot.png", 98, 900, 962, 962/3, mask="auto")

    # Weekly Performance
    pdf.setFillColor(colors.black)
    pdf.setFont("Inter-Bold", 28)
    pdf.drawString(154, 772,  "Weekly Performance")

    mcap_change = float(mcap_changes.loc[0,"mcap_percentage_change"])
    if mcap_change > 0:
        r, g, b = hex_to_rgb("#568475")  #green
        pdf.setFillColorRGB(r, g, b)
        pdf.drawString(670, 772,  f'+{mcap_change}%')
    if mcap_change < 0:
        r, g, b = hex_to_rgb("#D53E50")  #Orangey-red
        pdf.setFillColorRGB(r, g, b)
        pdf.drawString(670, 772,  f'{mcap_change}%')
    if mcap_change == 0:
        pdf.setFillColor(colors.black)
        pdf.drawString(670, 772,  f'{mcap_change}%')

    pdf.setFillColor(colors.black)
    pdf.drawString(835, 772,  f"IDR {format_number_short_2d(mcap_changes['mcap_start'])}")


    # Top Gainers
    for i in range(0,5):
        image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_gainers_losers['symbol'][i][0:4]}.webp").content))
        pdf.drawImage(image, 180, height - 990 - (88*i), 50, 50,mask="auto")

    for i in range (0,5):
        # Title
        pdf.setFillColor(colors.black)
        pdf.setFont("Inter-Bold", 32)
        pdf.drawString(260, height - 980 - (88*i), top_gainers_losers["symbol"][i][0:4])

    for i in range (0,5):
        # Title
        r, g, b = hex_to_rgb("#568475")  #green
        pdf.setFillColorRGB(r, g, b)
        pdf.setFont("Inter-Bold", 28)
        text = f"{top_gainers_losers['mcap_change_pct'][i]}%"
        text_width = pdf.stringWidth(text, 'Inter-Bold', 28)
        pdf.drawString(1200-text_width-600, height - 962 - (88*i),  text)

    for i in range (0,5):
        # Title
        pdf.setFillColor(colors.black)
        pdf.setFont("Inter", 20)
        text = f"IDR {format_number_short_1d(top_gainers_losers['mcap_end'][i])}"
        text_width = pdf.stringWidth(text, 'Inter', 20)
        pdf.drawString(1200-text_width-600, height - 990 - (88*i), text)

    # Top Losers
    for i in range(0,5):
        image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_gainers_losers['symbol'][i+5][0:4]}.webp").content))
        pdf.drawImage(image, 705, height - 990 - (88*i), 50, 50, mask="auto")

    for i in range (0,5):
        # Title
        pdf.setFillColor(colors.white)
        pdf.setFont("Inter-Bold", 32)
        pdf.drawString(770, height - 980 - (88*i), top_gainers_losers["symbol"][i+5][0:4])

    for i in range (0,5):
        # Title
        r, g, b = hex_to_rgb("#D53E50")  #Orangey-red
        pdf.setFillColorRGB(r, g, b)
        pdf.setFont("Inter-Bold", 28)
        text = f"{top_gainers_losers['mcap_change_pct'][i+5]}%"
        text_width = pdf.stringWidth(text, 'Inter-Bold', 28)
        pdf.drawString(1200-text_width-150, height - 962 - (88*i), text)

    for i in range (0,5):
        # Title
        pdf.setFillColor(colors.white)
        pdf.setFont("Inter", 20)
        text = f"IDR {format_number_short_1d(top_gainers_losers['mcap_end'][i+5])}"
        text_width = pdf.stringWidth(text, 'Inter', 20)
        pdf.drawString(1200-text_width-150, height - 990 - (88*i), text)

    # Page 2
    pdf.showPage()

    pdf.drawImage('asset/page/page_2.png', 0, 0, width, height)

    # Indices Performance
    for i in range(0, 3):
        # Calculate column center first
        center = 270 + (335 * i)
        y_percentage = height - 345
        y_price = height - 397

        # Retrieve values
        percentage = indices_changes.loc[i, 'price_change_pct']
        price = f"IDR {indices_changes.loc[i, 'end_price']}"

        # Format percentage text
        percentage_text = f"+{percentage}%" if percentage > 0 else f"{percentage}%"

        # Calculate width for centering
        percentage_width = pdf.stringWidth(percentage_text, "Inter-Bold", 55)
        price_width = pdf.stringWidth(price, "Inter", 36)

        percentage_x = center - (percentage_width / 2)
        price_x = center - (price_width / 2)

        # Set color and draw percentage first
        if percentage > 0:
            r, g, b = hex_to_rgb("#568475")  # green
            pdf.setFillColorRGB(r, g, b)
        elif percentage < 0:
            r, g, b = hex_to_rgb("#D53E50")  # orange-red
            pdf.setFillColorRGB(r, g, b)
        else:
            pdf.setFillColor(colors.black)

        pdf.setFont("Inter-Bold", 55)
        pdf.drawString(percentage_x, y_percentage, percentage_text)

        # Reset color for price
        pdf.setFillColor(colors.black)
        pdf.setFont("Inter", 36)
        pdf.drawString(price_x, y_price, price)


    # Top 3 Sectors
    for i in range(0,3):
        pdf.drawImage(f'asset/sectors/{sectors_changes.loc[i,"sector"]}.png', 139 + (i*330), 900, 45,45, mask="auto")
        pdf.setFillColor(colors.white)
        pdf.setFont("Inter-Bold", 24)
        draw_shrinking_text(pdf, sectors_changes.loc[i,'sub_sector'], 206, 203 + (i*330), 925, font_name='Inter-Bold', initial_font_size=24, min_font_size=12, color=colors.white)
        # pdf.drawString(203+ (i*330), 925, sectors_changes.loc[i,'sub_sector'])
        pdf.setFont("Inter", 20)
        pdf.drawString(203+ (i*330), 900, f"IDR {format_number_short_2d(sectors_changes.loc[i,'total_market_cap'])}")

    for i in range(0,3):
        #1w
        pdf.setFont("Inter", 18)
        pdf.setFillColor(colors.white)
        onew_x = 155 + (i * 330)
        onew_y = 850
        pdf.drawString(onew_x, onew_y, "1 Week")
        # Calculate the width of the YTD label
        onew_width = pdf.stringWidth("1 Week", "Inter", 18)

        # Center point for YTD
        onew_center = 155 + (i * 330) + (onew_width / 2)

        # now compute the width of the percentage text
        percentage = sectors_changes.loc[i,'mcap_change_1w']

        if percentage > 0:
            r, g, b = hex_to_rgb("#ABDDA4")  # green
            pdf.setFillColorRGB(r, g, b)
            text = f"+{percentage}%"
        elif percentage < 0:
            r, g, b = hex_to_rgb("#D53E50")  # orange-red
            pdf.setFillColorRGB(r, g, b)
            text = f"{percentage}%"
        else:
            pdf.setFillColor(colors.white)
            text = "0.00%"    

        # Measure text width for centering
        text_width = pdf.stringWidth(text, "Inter-Bold", 18)
        centered_x = onew_center - (text_width/2)

        # Now draw the percentage text centered under YTD
        percentage_y = 825
        pdf.setFont("Inter-Bold", 18)
        pdf.drawString(centered_x, percentage_y, text)

        #ytd
        pdf.setFont("Inter", 14)
        pdf.setFillColor(colors.white)
        ytd_x = 273 + (i * 330)
        ytd_y = 850
        pdf.drawString(ytd_x, ytd_y, "YTD")
        # Calculate the width of the YTD label
        ytd_width = pdf.stringWidth("YTD", "Inter", 14)

        # Center point for YTD
        ytd_center = 273 + (i * 330) + (ytd_width / 2)

        # now compute the width of the percentage text
        percentage = sectors_changes.loc[i,'mcap_change_ytd']

        if percentage > 0:
            r, g, b = hex_to_rgb("#ABDDA4")  # green
            pdf.setFillColorRGB(r, g, b)
            text = f"+{percentage}%"
        elif percentage < 0:
            r, g, b = hex_to_rgb("#D53E50")  # orange-red
            pdf.setFillColorRGB(r, g, b)
            text = f"{percentage}%"
        else:
            pdf.setFillColor(colors.white)
            text = "0.00%"    

        # Measure text width for centering
        text_width = pdf.stringWidth(text, "Inter-Bold", 14)
        centered_x = ytd_center - (text_width/2)

        # Now draw the percentage text centered under YTD
        percentage_y = 830
        pdf.setFont("Inter-Bold", 14)
        pdf.drawString(centered_x, percentage_y, text)

        #1y
        pdf.setFont("Inter", 14)
        pdf.setFillColor(colors.white)
        oney_x = 345 + (i * 330)
        oney_y = 850
        pdf.drawString(oney_x, oney_y, "1 Year")
        # Calculate the width of the YTD label
        oney_width = pdf.stringWidth("1 Year", "Inter", 14)

        # Center point for 1y
        oney_center = 345 + (i * 330) + (oney_width / 2)

        # now compute the width of the percentage text
        percentage = sectors_changes.loc[i,'mcap_change_1y']

        if percentage > 0:
            r, g, b = hex_to_rgb("#ABDDA4")  # green
            pdf.setFillColorRGB(r, g, b)
            text = f"+{percentage}%"
        elif percentage < 0:
            r, g, b = hex_to_rgb("#D53E50")  # orange-red
            pdf.setFillColorRGB(r, g, b)
            text = f"{percentage}%"
        else:
            pdf.setFillColor(colors.white)
            text = "0.00%"    

        # Measure text width for centering
        text_width = pdf.stringWidth(text, "Inter-Bold", 14)
        centered_x = oney_center - (text_width/2)

        # Now draw the percentage text centered under YTD
        percentage_y = 830
        pdf.setFont("Inter-Bold", 14)
        pdf.drawString(centered_x, percentage_y, text)

    for j in range (0,3):
        for i in range (0,3):
            image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_3_comp_sectors.loc[i + (j*3),'symbol'][0:4]}.webp").content))
            pdf.drawImage(image, 162 + (j*330), 756 - (i*75), 26,26, mask="auto")
            pdf.setFont("Inter", 18)
            pdf.setFillColor(colors.white)
            pdf.drawString(154 + (j*330) ,734 - (i*75), top_3_comp_sectors.loc[i + (j*3),"symbol"][0:4])

            center = 260 + (j * 330)
            y_percentage = 764 - (i * 75)
            y_idr = 734 - (i * 75)

            # Retrieve values
            percentage = top_3_comp_sectors.loc[i + (j*3), "mcap_change_pct"]
            percentage_text = f"+{percentage}%" if percentage > 0 else f"{percentage}%"

            text = f"IDR {format_number_short_1d(top_3_comp_sectors.loc[i + (j*3), 'close'])}"

            # Calculate width for centering
            percentage_width = pdf.stringWidth(percentage_text, "Inter-Bold", 22)
            text_width = pdf.stringWidth(text, "Inter", 18)

            percentage_x = center - (percentage_width/2)
            text_x = center - (text_width/2)

            # Set color for percentage first
            if percentage > 0:
                r, g, b = hex_to_rgb("#ABDDA4")  # green
                pdf.setFillColorRGB(r, g, b)
            elif percentage < 0:
                r, g, b = hex_to_rgb("#D53E50")  # orange-red
                pdf.setFillColorRGB(r, g, b)
            else:
                pdf.setFillColor(colors.white)

            # Draw percentage
            pdf.setFont("Inter-Bold", 22)
            pdf.drawString(percentage_x, y_percentage, percentage_text)

            # Reset color for the secondary text
            pdf.setFillColor(colors.white)
            pdf.setFont("Inter", 18)
            pdf.drawString(text_x, y_idr, text)
                    

            # Inside your for loop:

            center = 358 + (j * 330)
            y_label = 760 - (i * 75)
            y_number = 740 - (i * 75)

            # Texts to draw
            label = "P/E"
            if top_3_comp_sectors.loc[i + (j*3), 'round'] > 100:
                number = "> 100"
            elif top_3_comp_sectors.loc[i + (j*3), 'round'] < 0:
                number = "< 0"
            else:
                number = f"{top_3_comp_sectors.loc[i + (j*3), 'round']}"
            
            # Calculate their widths
            label_width = pdf.stringWidth(label, "Inter", 18)
            number_width = pdf.stringWidth(number, "Inter-Bold", 16)

            # Center them
            label_x = center - (label_width/2)
            number_x = center - (number_width/2)

            # Set color and font
            pdf.setFillColor(colors.white)
            pdf.setFont("Inter", 18)

            # Draw
            pdf.drawString(label_x, y_label, label)

            pdf.setFillColor(colors.white)
            pdf.setFont("Inter-Bold", 16)
            pdf.drawString(number_x, y_number, number)

    # Top volume Traded
    image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_volume.loc[0,'symbol'][0:4]}.webp").content))
    pdf.drawImage(image, 158, 429-110, 63,63, mask="auto")
    pdf.setFont("Inter", 40)
    pdf.setFillColor(colors.white)
    pdf.drawString(244, 337, top_volume.loc[0,"symbol"][0:4])
    pdf.setFont("Inter-Semi-Bold", 40)
    pdf.drawString(430, 337, format_number_short_2d(top_volume.loc[0,"total_volume"]))

    for i in range(0,4):
        # Center base
        center = 193 + (i * 105)  # 193 = 168 + 25 (half logo width)

        # Draw logo first, centered
        image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_volume.loc[i+1,'symbol'][0:4]}.webp").content))
        pdf.drawImage(image, center - 25, 240, 50, 50, mask='auto')

        # Draw symbol, centered
        symbol = top_volume.loc[i+1,"symbol"][0:4]
        symbol_width = pdf.stringWidth(symbol, "Inter", 26)
        symbol_x = center - (symbol_width/2)
        symbol_y = 205

        pdf.setFont("Inter", 26)
        pdf.setFillColor(colors.white)
        pdf.drawString(symbol_x, symbol_y, symbol)

        # Now draw number, centered
        number = format_number_short_2d(top_volume.loc[i+1,"total_volume"])

        number_width = pdf.stringWidth(number, "Inter-Semi-Bold", 26)
        number_x = center - (number_width/2)
        number_y = 170

        pdf.setFont("Inter-Semi-Bold", 26)
        pdf.drawString(number_x, number_y, number)

    # Top value Traded
    image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_value.loc[0,'symbol'][0:4]}.webp").content))
    pdf.drawImage(image, 658, 429-110, 63,63, mask="auto")
    pdf.setFont("Inter", 40)
    pdf.setFillColor(colors.white)
    pdf.drawString(744, 337, top_value.loc[0,"symbol"][0:4])
    pdf.setFont("Inter-Semi-Bold", 40)
    pdf.drawString(930, 337, format_number_short_2d(top_value.loc[0,"total_value"]))

    for i in range(0,4):
        # Center base
        center = 668 + (i * 105) + 25  # 668 + (i*105) is the left corner, 25 is half of logo width (50/2)

        # Draw logo first, centered
        image = ImageReader(BytesIO(requests.get(f"https://storage.googleapis.com/sectorsapp/logo/{top_value.loc[i+1,'symbol'][0:4]}.webp").content))
        pdf.drawImage(image, center - 25, 240, 50, 50, mask='auto')

        # Draw symbol, centered
        symbol = top_value.loc[i+1,"symbol"][0:4]
        symbol_width = pdf.stringWidth(symbol, "Inter", 26)
        symbol_x = center - (symbol_width/2)
        symbol_y = 205

        pdf.setFont("Inter", 26)
        pdf.setFillColor(colors.white)
        pdf.drawString(symbol_x, symbol_y, symbol)

        # Now draw number, centered
        number = format_number_short_2d(top_value.loc[i+1,"total_value"])

        number_width = pdf.stringWidth(number, "Inter-Semi-Bold", 26)
        number_x = center - (number_width/2)
        number_y = 170

        pdf.setFont("Inter-Semi-Bold", 26)
        pdf.drawString(number_x, number_y, number)

    # Page 3
    pdf.showPage()
    ## Calendar
    unfill_cal_box = "asset/calendar_asset/unfill_cal.png"
    fill_cal_box = "asset/calendar_asset/fill_cal.png"
    ipo_num = "asset/calendar_asset/ipo_cal.png"
    div_num = "asset/calendar_asset/div_cal.png"
    split_num = "asset/calendar_asset/split_cal.png"
    pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)

    pdf.setFont("Inter-Bold", 36)
    pdf.setFillColor(colors.white)
    pdf.drawString(109, 1240, date_generator("calendar"))

    for i in range (0,5):
        if ca_comp.iloc[i]["total"] == 0:
            pdf.drawImage(unfill_cal_box, 109+(201*i), 1080, 180,128, mask="auto")
        else:
            pdf.drawImage(fill_cal_box, 109+(201*i), 1080, 180,128, mask="auto")
        
        pdf.setFont("Inter-Bold", 48)
        pdf.setFillColor(colors.white)
        pdf.drawString(130+(201*i), 1143, f"{ca_comp.loc[i,'date'].day}")
        pdf.setFont("Inter", 24)
        pdf.setFillColor(colors.white)
        pdf.drawString(130+(201*i), 1110, ca_comp.iloc[i]['date'].strftime('%a'))

        # calendar numbering
        fields = [
        ("ipo", ipo_num),
        ("div", div_num),
        ("split", split_num)
        ]

        # Filter and collect only non-zero fields for current row
        active_items = [(name, icon) for name, icon in fields if ca_comp.iloc[i][name] != 0]

        # Set vertical position start points depending on how many fields are shown
        base_y = 1165 if len(active_items) == 3 else 1150 if len(active_items) == 2 else 1130

        for index, (name, icon) in enumerate(active_items):
            y = base_y - (index * 35)  # adjust vertical position with spacing
            x = 233 + (201 * i)

            # Draw image
            pdf.drawImage(icon, x, y, 36, 30, mask="auto")

            # Set text properties
            pdf.setFont("Inter-Bold", 20)
            r, g, b = hex_to_rgb("#8B004C")
            pdf.setFillColorRGB(r, g, b)

            # Get and center the text
            text = str(ca_comp.iloc[i][name])
            text_width = pdf.stringWidth(text, "Inter-Bold", 20)
            pdf.drawString(x + 18 - text_width / 2, y + 15 - 10 + 3, text)  # center text

    ## CA Detail
    event_sums = ca_comp.set_index('date').sum()
    ipo_sum = df_ipo.shape[0]
    split_sum = stock_split.shape[0]
    div_sum = df_div.shape[0]

    image_x = 109
    image_width = 982
    image_center_x = image_x + image_width / 2

    # CA Detail Content
    def ca_detail(df, label, i, y_pos):

        def create_ca_date_and_stock(y_pos, day, month, stock, label):
            font_name = "Inter-Bold"
            x_pos_text = 124
            y_pos_text = y_pos + 34

            # Measure text widths
            text_above_width = stringWidth(month, font_name, 20)
            text_center_width = stringWidth(day, font_name, 40)

            # Calculate new x position to center "8" under "Apr"
            x_center = x_pos_text + (text_above_width - text_center_width) / 2

            # Draw both texts

            pdf.setFillColor(colors.white)
            pdf.setFont(font_name, 20)
            pdf.drawString(x_pos_text, y_pos_text, month)

            pdf.setFont(font_name, 40)
            pdf.drawString(x_center, y_pos_text+25, day)

            # Stock & Logo
            pdf.setFont("Inter-Bold", 32)
            pdf.setFillColor(colors.black)

            if label == "IPO Listing":
                pdf.drawString(215, y_pos+52,stock)
            else:
                pdf.drawString(261, y_pos+52,stock)
                pdf.drawImage(f"https://storage.googleapis.com/sectorsapp/logo/{stock}.webp", 196, y_pos+38, 49,49, mask="auto")

        if label == "Dividend and Upcoming Dividend":
            create_ca_date_and_stock(y_pos, f"{df.iloc[i]['ex_date'].day}",f"{df.iloc[i]['ex_date'].strftime('%b')}",df.iloc[i]['symbol'][0:4], label)

            pdf.setFont("Inter-Bold", 32)
            pdf.setFillColor(colors.black)
            pdf.drawString(690, y_pos+65,f"IDR {df.iloc[i]['dividend_amount']:,.0f}")

            pdf.setFont("Inter", 24)
            pdf.setFillColor(colors.black)
            pdf.drawString(690, y_pos+35,f"Yield {df.iloc[i]['dividend_yield']}%")

            pdf.setFont("Inter", 24)
            pdf.setFillColor(colors.black)
            pdf.drawString(891, y_pos+70,"Payment")

            pdf.setFont("Inter-Bold", 32)
            pdf.setFillColor(colors.black)
            pdf.drawString(891, y_pos+35,f"{df.iloc[i]['payment_date'].strftime('%d %b %y')}")
                
        elif label == "Stock Splitting":
            create_ca_date_and_stock(y_pos, f"{df.iloc[i]['date'].day}",f"{df.iloc[i]['date'].strftime('%b')}",df.iloc[i]['symbol'][0:4], label)

            pdf.setFont("Inter", 24)
            pdf.setFillColor(colors.black)
            pdf.drawString(891, y_pos+80,"Split Ratio")

            pdf.setFont("Inter-Bold", 32)
            pdf.setFillColor(colors.black)
            pdf.drawString(891, y_pos+35,f"{df.iloc[i]['split_ratio']}-for-1")

        elif label == "IPO Listing":
            create_ca_date_and_stock(y_pos, f"{df.iloc[i]['offering_end_date'].day}",f"{df.iloc[i]['offering_end_date'].strftime('%b')}",df.iloc[i]['symbol'][0:4], label)

            pdf.setFont("Inter-Bold", 32)
            pdf.setFillColor(colors.black)
            pdf.drawString(680, y_pos+65,f"IDR {df.iloc[i]['offering_price']}")

            pdf.setFont("Inter", 24)
            pdf.setFillColor(colors.black)
            pdf.drawString(680, y_pos+35,f"Shares {format_number_short_1d(df.iloc[i]['shares_offered'])}")

            pdf.setFont("Inter", 24)
            pdf.setFillColor(colors.black)
            pdf.drawString(891, y_pos+70,"Listing")

            pdf.setFont("Inter-Bold", 32)
            pdf.setFillColor(colors.black)
            pdf.drawString(891, y_pos+35,f"{df.iloc[i]['listing_date'].strftime('%d %b %y')}")

    # One CA in One Week
    def draw_event_section(pdf, label, asset_path, count, first_page_offset, next_page_offset):
        pdf.setFont("Inter-Bold", 36)
        pdf.setFillColor(colors.white)

        # Text alignment setup
        text_width = stringWidth(label, "Inter-Bold", 36)
        text_x = image_center_x - text_width / 2

        # === Page 1: draw up to 6 items ===
        text_y = first_page_offset
        pdf.drawString(text_x, text_y, label)

        items_on_first_page = min(count, 6)
        for i in range(items_on_first_page):
            y_pos = first_page_offset - 40 - 124 - ((124 + 25) * i)
            pdf.drawImage(asset_path, image_x, y_pos, 982, 124, mask="auto")

            if label == "Dividend and Upcoming Dividend":
                ca_detail(df_div, label, i, y_pos)
                
                
            elif label == "Stock Splitting":
                ca_detail(stock_split, label, i, y_pos)
                

            elif label == "IPO Listing":
                ca_detail(df_ipo, label, i, y_pos)
                

        remaining = count - items_on_first_page

        if remaining > 0:
            # Number of full additional pages needed (7 items per page)
            full_pages = (remaining - 1) // 7 + 1

            for page in range(full_pages):
                pdf.showPage()
                pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                pdf.setFont("Inter-Bold", 36)
                pdf.setFillColor(colors.white)

                text_y = next_page_offset
                pdf.drawString(text_x, text_y, label)

                for i in range(7):
                    global_index = items_on_first_page + page * 7 + i
                    if global_index >= count:
                        break
                    y_pos = next_page_offset - 40 - 124 - ((124 + 25) * i)
                    pdf.drawImage(asset_path, image_x, y_pos, 982, 124, mask="auto")

                    if label == "Dividend and Upcoming Dividend":
                        ca_detail(df_div, label, i+items_on_first_page+(7*page), y_pos)
                        
                        
                    elif label == "Stock Splitting":
                        ca_detail(stock_split, label, i+items_on_first_page+(7*page), y_pos)
                        

                    elif label == "IPO Listing":
                        ca_detail(df_ipo, label, i+items_on_first_page+(7*page), y_pos)
        
        return y_pos

    # Two CA in One Week Function
    def draw_two_event_sections(
        pdf,
        first_label, first_asset_path, first_count,
        second_label, second_asset_path, second_count,
        first_page_offset, next_page_offset
    ):
        
        first_final_content_y_pos = draw_event_section(pdf, first_label, first_asset_path, first_count,
                        first_page_offset, next_page_offset)
        
        if first_count < 5:
            # Stack split section after IPO
            y_last_ipo = first_page_offset - 40 - 124 - ((124 + 25) * (first_count - 1))
            y_split = y_last_ipo - 43 -70 # 43 px padding

            pdf.setFont("Inter-Bold", 36)
            pdf.setFillColor(colors.white)
            text = second_label
            text_width = stringWidth(text, "Inter-Bold", 36)
            text_x = image_center_x - text_width / 2
            pdf.drawString(text_x, y_split, text)
            
            for i in range(min(second_count,5-first_count)):
                y_pos = y_split - 40 - 124 - ((124 + 25) * i)
                pdf.drawImage(second_asset_path, image_x, y_pos, 982, 124, mask="auto")

                if second_label == "Dividend and Upcoming Dividend":
                    ca_detail(df_div, second_label, i, y_pos)
                    
                    
                elif second_label == "Stock Splitting":
                    ca_detail(stock_split, second_label, i, y_pos)
                    

                elif second_label == "IPO Listing":
                    ca_detail(df_ipo, second_label, i, y_pos)

            remaining = second_count - (5-first_count)

            full_pages = (remaining - 1) // 7 + 1
            for page in range(full_pages):
                pdf.showPage()
                pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                pdf.setFont("Inter-Bold", 36)
                pdf.setFillColor(colors.white)

                text_y = next_page_offset
                pdf.drawString(text_x, text_y, text)

                for i in range(7):
                    if remaining <= 0:
                        break
                    y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                    pdf.drawImage(second_asset_path, image_x, y_pos, 982, 124, mask="auto")

                    if second_label == "Dividend and Upcoming Dividend":
                        ca_detail(df_div, second_label, i+min(second_count,5-first_count)+(page*7), y_pos)
                        
                        
                    elif second_label == "Stock Splitting":
                        ca_detail(stock_split, second_label, i+min(second_count,5-first_count)+(page*7), y_pos)
                        

                    elif second_label == "IPO Listing":
                        ca_detail(df_ipo, second_label, i+min(second_count,5-first_count)+(page*7), y_pos)

                    remaining -= 1
        
        elif first_count == 5 or first_count == 6:
            remaining = second_count
            full_pages = (remaining - 1) // 7 + 1
            for page in range(full_pages):
                pdf.showPage()
                pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                pdf.setFont("Inter-Bold", 36)
                pdf.setFillColor(colors.white)

                pdf.setFont("Inter-Bold", 36)
                pdf.setFillColor(colors.white)
                text = second_label
                text_width = stringWidth(text, "Inter-Bold", 36)
                text_x = image_center_x - text_width / 2

                text_y = next_page_offset
                pdf.drawString(text_x, text_y, text)

                for i in range(7):
                    if remaining <= 0:
                        break
                    y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                    pdf.drawImage(second_asset_path, image_x, y_pos, 982, 124, mask="auto")

                    if second_label == "Dividend and Upcoming Dividend":
                        ca_detail(df_div, second_label, i+(page*7), y_pos)
                        
                        
                    elif second_label == "Stock Splitting":
                        ca_detail(stock_split, second_label, i+(page*7), y_pos)
                        

                    elif second_label == "IPO Listing":
                        ca_detail(df_ipo, second_label, i+(page*7), y_pos)

                    remaining -= 1

        elif first_count > 6:
            y_split = first_final_content_y_pos - 43 -70 # 43 px padding

            pdf.setFont("Inter-Bold", 36)
            pdf.setFillColor(colors.white)
            text = second_label
            text_width = stringWidth(text, "Inter-Bold", 36)
            text_x = image_center_x - text_width / 2
            pdf.drawString(text_x, y_split, text)
            
            for i in range(min(second_count,6-(first_count-6))):
                y_pos = y_split - 40 - 124 - ((124 + 25) * i)
                pdf.drawImage(second_asset_path, image_x, y_pos, 982, 124, mask="auto")

                if second_label == "Dividend and Upcoming Dividend":
                    ca_detail(df_div, second_label, i, y_pos)
                    
                    
                elif second_label == "Stock Splitting":
                    ca_detail(stock_split, second_label, i, y_pos)
                    

                elif second_label == "IPO Listing":
                    ca_detail(df_ipo, second_label, i, y_pos)

            remaining = second_count - (6-(first_count-6))

            full_pages = (remaining - 1) // 7 + 1
            for page in range(full_pages):
                pdf.showPage()
                pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                pdf.setFont("Inter-Bold", 36)
                pdf.setFillColor(colors.white)

                text_y = next_page_offset
                pdf.drawString(text_x, text_y, text)

                for i in range(7):
                    if remaining <= 0:
                        break
                    y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                    pdf.drawImage(second_asset_path, image_x, y_pos, 982, 124, mask="auto")

                    if second_label == "Dividend and Upcoming Dividend":
                        ca_detail(df_div, second_label, i+min(second_count,6-(first_count-6))+(page*7), y_pos)
                        
                        
                    elif second_label == "Stock Splitting":
                        ca_detail(stock_split, second_label, i+min(second_count,6-(first_count-6))+(page*7), y_pos)
                        

                    elif second_label == "IPO Listing":
                        ca_detail(df_ipo, second_label, i+min(second_count,6-(first_count-6))+(page*7), y_pos)

                    remaining -= 1
        
        return y_pos

    # --- Condition 1: Only Dividends ---
    if ipo_sum == 0 and split_sum == 0 and div_sum != 0:
        draw_event_section(pdf, "Dividend and Upcoming Dividend", "asset/calendar_asset/dividend.png", div_sum,
                        1500 - 483 - 20, 1500 - 257 - 20)

    # --- Condition 2: Only IPOs ---
    elif ipo_sum != 0 and split_sum == 0 and div_sum == 0:
        draw_event_section(pdf, "IPO Listing", "asset/calendar_asset/ipo_long.png", ipo_sum,
                        1500 - 483 - 20, 1500 - 257 - 20)

    # --- Condition 3: Only Splits ---
    elif ipo_sum == 0 and split_sum != 0 and div_sum == 0:
        draw_event_section(pdf, "Stock Splitting", "asset/calendar_asset/split_long.png", split_sum,
                        1500 - 483 - 20, 1500 - 257 - 20)
        
    elif ipo_sum != 0 and split_sum != 0 and div_sum == 0:
        draw_two_event_sections(
        pdf,
        "IPO Listing", "asset/calendar_asset/ipo_long.png", ipo_sum,
        "Stock Splitting", "asset/calendar_asset/split_long.png", split_sum,
        1500 - 483 - 20, 1500 - 257 - 20
    )

    elif ipo_sum == 0 and split_sum != 0 and div_sum != 0:
        draw_two_event_sections(
        pdf,
        "Stock Splitting", "asset/calendar_asset/split_long.png", split_sum,
        "Dividend and Upcoming Dividend", "asset/calendar_asset/dividend.png", div_sum,
        1500 - 483 - 20, 1500 - 257 - 20
    )

    elif ipo_sum != 0 and split_sum == 0 and div_sum != 0:
        draw_two_event_sections(
        pdf,
        "IPO Listing", "asset/calendar_asset/ipo_long.png", ipo_sum,
        "Dividend and Upcoming Dividend", "asset/calendar_asset/dividend.png", div_sum,
        1500 - 483 - 20, 1500 - 257 - 20
    )
        
    elif ipo_sum != 0 and split_sum != 0 and div_sum != 0:

        # # --- IPO Section ---
        ca_num = ipo_sum + split_sum + div_sum
        
        y_last_ipo = draw_two_event_sections(
        pdf,
        "IPO Listing", "asset/calendar_asset/ipo_long.png", ipo_sum,
        "Stock Splitting", "asset/calendar_asset/split_long.png", split_sum,
        1500 - 483 - 20, 1500 - 257 - 20
    )

        if ca_num > 4:
            remaining = split_sum - (5-ipo_sum)
            last_page_num = 6 - remaining

            if last_page_num == 0:
                remaining = div_sum
                
                full_pages = (remaining - 1) // 7 + 1

                for page in range(full_pages):
                    pdf.showPage()
                    pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                    pdf.setFont("Inter-Bold", 36)
                    pdf.setFillColor(colors.white)

                    pdf.setFont("Inter-Bold", 36)
                    pdf.setFillColor(colors.white)
                    text = "Dividend and Upcoming Dividend"
                    text_width = stringWidth(text, "Inter-Bold", 36)
                    text_x = image_center_x - text_width / 2

                    text_y = 1500 - 257 - 20
                    pdf.drawString(text_x, text_y, text)

                    for i in range(7):
                        if remaining <= 0:
                            break
                        y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                        pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                        ca_detail(df_div, "Dividend and Upcoming Dividend", i+(page*7), y_pos)

                        remaining -= 1

            elif last_page_num != 0:
                if ipo_sum + split_sum >= 5: 
                    y_split = y_last_ipo - 43 -70 # 43 px padding

                    pdf.setFont("Inter-Bold", 36)
                    pdf.setFillColor(colors.white)
                    text = "Dividend and Upcoming Dividend"
                    text_width = stringWidth(text, "Inter-Bold", 36)
                    text_x = image_center_x - text_width / 2
                    pdf.drawString(text_x, y_split, text)
                    
                    for i in range(min(div_sum,last_page_num)):
                        y_pos = y_split - 40 - 124 - ((124 + 25) * i)
                        pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                        ca_detail(df_div, "Dividend and Upcoming Dividend", i, y_pos)

                    remaining = div_sum - min(div_sum,last_page_num)

                    full_pages = (remaining - 1) // 7 + 1

                    for page in range(full_pages):
                        pdf.showPage()
                        pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                        pdf.setFont("Inter-Bold", 36)
                        pdf.setFillColor(colors.white)

                        text_y = 1500 - 257 - 20
                        pdf.drawString(text_x, text_y, text)

                        for i in range(7):
                            if remaining <= 0:
                                break
                            y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                            pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                            ca_detail(df_div, "Dividend and Upcoming Dividend", i+(page*7)+min(div_sum,last_page_num), y_pos)

                            remaining -= 1
                
                elif ipo_sum + split_sum == 4:
                    y_split = y_last_ipo - 43 -70 # 43 px padding

                    full_pages = (div_sum - 1) // 7 + 1

                    for page in range(full_pages):
                        pdf.showPage()
                        pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                        pdf.setFont("Inter-Bold", 36)
                        pdf.setFillColor(colors.white)
                        text = "Dividend and Upcoming Dividend"
                        text_width = stringWidth(text, "Inter-Bold", 36)
                        text_x = image_center_x - text_width / 2

                        text_y = 1500 - 257 - 20
                        pdf.drawString(text_x, text_y, text)

                        remaining = div_sum
                        for i in range(7):
                            if remaining <= 0:
                                break
                            y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                            pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                            ca_detail(df_div, "Dividend and Upcoming Dividend", i+(page*7), y_pos)

                            remaining -= 1

                elif ipo_sum + split_sum < 4:
                    y_split = y_last_ipo - 43 -70 # 43 px padding

                    pdf.setFont("Inter-Bold", 36)
                    pdf.setFillColor(colors.white)
                    text = "Dividend and Upcoming Dividend"
                    text_width = stringWidth(text, "Inter-Bold", 36)
                    text_x = image_center_x - text_width / 2
                    pdf.drawString(text_x, y_split, text)
                    
                    for i in range(min(div_sum,4-ipo_sum - split_sum)):
                        y_pos = y_split - 40 - 124 - ((124 + 25) * i)
                        pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                        ca_detail(df_div, "Dividend and Upcoming Dividend", i, y_pos)

                    remaining = div_sum - min(div_sum,4-ipo_sum - split_sum)

                    full_pages = (remaining - 1) // 7 + 1

                    for page in range(full_pages):
                        pdf.showPage()
                        pdf.drawImage('asset/page/page_3.png', 0, 0, width, height)
                        pdf.setFont("Inter-Bold", 36)
                        pdf.setFillColor(colors.white)

                        text_y = 1500 - 257 - 20
                        pdf.drawString(text_x, text_y, text)

                        for i in range(7):
                            if remaining <= 0:
                                break
                            y_pos = text_y - 40 - 124 - ((124 + 25) * i)
                            pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                            ca_detail(df_div, "Dividend and Upcoming Dividend", i+(page*7)+min(div_sum,4-ipo_sum - split_sum), y_pos)

                            remaining -= 1

        elif ca_num <= 4:
            y_split = y_last_ipo - 43 -70 # 43 px padding

            pdf.setFont("Inter-Bold", 36)
            pdf.setFillColor(colors.white)
            text = "Dividend and Upcoming Dividend"
            text_width = stringWidth(text, "Inter-Bold", 36)
            text_x = image_center_x - text_width / 2
            pdf.drawString(text_x, y_split, text)
            
            for i in range(div_sum):
                y_pos = y_split - 40 - 124 - ((124 + 25) * i)
                pdf.drawImage("asset/calendar_asset/dividend.png", image_x, y_pos, 982, 124, mask="auto")
                ca_detail(df_div, "Dividend and Upcoming Dividend", i, y_pos)

            
    # CTA page
    pdf.showPage()
    pdf.drawImage('asset/page/cta.png', 0, 0, width, height)

    # Save pdf
    pdf.showPage()
    pdf.save()

    print("✅ PDF created")

def main():
    # Setting Connection
    cur = set_connection()

    # Page 1 Data
    ## 1 Week historical mcap data
    hist_mcap = fetch_query("""
        SELECT sum(market_cap) as market_cap, date
        FROM idx_daily_data
        WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
        AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
        group by date;""", cur)

    hist_mcap = full_week(hist_mcap)

    ## 1 Week market cap changes
    mcap_changes = fetch_query("""
                    SELECT
        -- Start of week market cap
        (SELECT SUM(market_cap)
        FROM idx_daily_data
        WHERE date = (
            SELECT MIN(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
            AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
        )
        ) AS mcap_start,

        -- End of week market cap
        (SELECT SUM(market_cap)
        FROM idx_daily_data
        WHERE date = (
            SELECT MAX(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
            AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
        )
        ) AS mcap_end,

        -- Difference
        (
            (SELECT SUM(market_cap)
            FROM idx_daily_data
            WHERE date = (
            SELECT MAX(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            )
            -
            (SELECT SUM(market_cap)
            FROM idx_daily_data
            WHERE date = (
            SELECT MIN(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            )
        ) AS mcap_difference,

        -- Percentage difference
        round((
            (
            (SELECT SUM(market_cap)
            FROM idx_daily_data
            WHERE date = (
                SELECT MAX(date)
                FROM idx_daily_data
                WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            )
            -
            (SELECT SUM(market_cap)
            FROM idx_daily_data
            WHERE date = (
                SELECT MIN(date)
                FROM idx_daily_data
                WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            )
            ) * 100.0
            /
            NULLIF((
            SELECT SUM(market_cap)
            FROM idx_daily_data
            WHERE date = (
                SELECT MIN(date)
                FROM idx_daily_data
                WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            ), 0)
        ),2) AS mcap_percentage_change;""", cur)
    
    ## 1 week top gainers and losers
    top_gainers_losers = fetch_query("""
        -- Top 3 gainers
        SELECT *
        FROM (
        SELECT
            start_data.symbol,
            start_data.mcap_start,
            end_data.mcap_end,
            end_data.mcap_end - start_data.mcap_start AS mcap_change,
            ROUND(100.0 * (end_data.mcap_end - start_data.mcap_start) / NULLIF(start_data.mcap_start, 0), 2) AS mcap_change_pct
        FROM (
            SELECT symbol, market_cap AS mcap_start,date
            FROM idx_daily_data
            WHERE date = (
            SELECT MIN(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
        ) AS start_data
        JOIN (
            SELECT symbol, market_cap AS mcap_end,date
            FROM idx_daily_data
            WHERE date = (
            SELECT MAX(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
        ) AS end_data
        ON start_data.symbol = end_data.symbol
        WHERE start_data.mcap_start >= 5000000000000
        ORDER BY mcap_change_pct DESC
        LIMIT 5
        ) AS top_gainers

        UNION ALL

        -- Bottom 3 losers
        SELECT *
        FROM (
        SELECT
            start_data.symbol,
            start_data.mcap_start,
            end_data.mcap_end,
            end_data.mcap_end - start_data.mcap_start AS mcap_change,
            ROUND(100.0 * (end_data.mcap_end - start_data.mcap_start) / NULLIF(start_data.mcap_start, 0), 2) AS mcap_change_pct
        FROM (
            SELECT symbol, SUM(market_cap) AS mcap_start
            FROM idx_daily_data
            WHERE date = (
            SELECT MIN(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            GROUP BY symbol
        ) AS start_data
        JOIN (
            SELECT symbol, SUM(market_cap) AS mcap_end
            FROM idx_daily_data
            WHERE date = (
            SELECT MAX(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
            GROUP BY symbol
        ) AS end_data
        ON start_data.symbol = end_data.symbol
        WHERE start_data.mcap_start >= 5000000000000
        ORDER BY mcap_change_pct ASC
        LIMIT 5
        ) AS bottom_losers;
                                        """, cur)

    top_gainers_losers["mcap_change_pct"] = top_gainers_losers["mcap_change_pct"].apply(lambda x: f"+{x}" if x>0 else x)

    # Page 2 Data

    ## Indices Performance
    indices_changes = fetch_query("""
        -- indices change
        SELECT
        start_data.index_code,
        start_data.price AS start_price,
        end_data.price AS end_price,
        ROUND((end_data.price - start_data.price)::numeric,2) AS price_change,
        ROUND(
            (100.0 * (end_data.price - start_data.price) / NULLIF(start_data.price, 0))::NUMERIC,
            2
        ) AS price_change_pct
        FROM
        (
            SELECT index_code, price
            FROM index_daily_data
            WHERE  date in (
            SELECT min(date)
            FROM index_daily_data
            WHERE 
            date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week' and index_code IN ('LQ45', 'IDXBUMN20', 'JII70')
            )
        ) AS start_data
        JOIN (
            SELECT index_code, price
            FROM index_daily_data
            WHERE date = (
            SELECT MAX(date)
            FROM index_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week' and index_code IN ('LQ45', 'IDXBUMN20', 'JII70')
            )
            AND index_code IN ('LQ45', 'IDXBUMN20', 'JII70')
        ) AS end_data
        ON start_data.index_code = end_data.index_code;
        """, cur)

    custom_order = ['IDXBUMN20', 'LQ45', 'JII70'] # make custom order
    indices_changes['index_code'] = pd.Categorical(indices_changes['index_code'], categories=custom_order, ordered=True)
    indices_changes = indices_changes.sort_values('index_code')

    ## Sectors Performance
    sectors_changes = fetch_query("""
        SELECT 
        sector,
        sub_sector,
        total_market_cap,
        ROUND((mcap_summary::jsonb->'mcap_change'->>'1w')::numeric * 100, 2) AS mcap_change_1w,
        ROUND((mcap_summary::jsonb->'mcap_change'->>'1y')::numeric * 100, 2) AS mcap_change_1y,
        ROUND((mcap_summary::jsonb->'mcap_change'->>'ytd')::numeric * 100, 2) AS mcap_change_ytd
        FROM idx_sector_reports
        order by abs((mcap_summary::jsonb->'mcap_change'->>'1w')::numeric * 100) desc
        limit 3;
        """, cur)

    sectors_changes["sector"] = (
        sectors_changes["sector"]
        .str.strip()
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)  # remove special characters
        .str.replace(r'\s+', '_', regex=True)     # replace spaces with underscores
    )

    # Top 3 companies per sector
    top_3_comp_sectors = fetch_query("""
        with daily_data as (SELECT changes.*, ism.sub_sector
        FROM (
        SELECT
            start_data.symbol,
            end_data.close,
            ROUND(100.0 * (end_data.mcap_end - start_data.mcap_start) / NULLIF(start_data.mcap_start, 0), 2) AS mcap_change_pct
        FROM (
            SELECT symbol, market_cap AS mcap_start,date
            FROM idx_daily_data
            WHERE date = (
            SELECT MIN(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
        ) AS start_data
        JOIN (
            SELECT symbol, market_cap AS mcap_end, close,date
            FROM idx_daily_data
            WHERE date = (
            SELECT MAX(date)
            FROM idx_daily_data
            WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
                AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
            )
        ) AS end_data
        ON start_data.symbol = end_data.symbol
        WHERE start_data.mcap_start >= 1000000000000
        ORDER BY mcap_change_pct DESC
        ) as changes
        left join idx_company_profile icp on changes.symbol = icp.symbol
        left join idx_subsector_metadata ism on icp.sub_sector_id = ism.sub_sector_id),
        daily_change as (SELECT *
        FROM (
        SELECT *,
                ROW_NUMBER() OVER (PARTITION BY sub_sector ORDER BY mcap_change_pct DESC) AS rn
        FROM daily_data
        WHERE sub_sector IN (
            SELECT sub_sector 
            FROM idx_sector_reports
            ORDER BY abs((mcap_summary::jsonb->'mcap_change'->>'1w')::numeric * 100) desc
            LIMIT 3
        )
        ) AS ranked
        WHERE rn <= 3
        ORDER BY sub_sector, mcap_change_pct DESC),
        sub_sec_rank as (
        SELECT 
        sub_sector,
        RANK() OVER (
            ORDER BY ABS((mcap_summary::jsonb->'mcap_change'->>'1w')::numeric * 100) DESC
        ) AS rank_sub_sec
        FROM idx_sector_reports
        LIMIT 3
        )
        select daily_change.*, round(idmd.pe_ttm::numeric,2), isr.total_market_cap from daily_change
        left join idx_calc_metrics_daily idmd on daily_change.symbol = idmd.symbol
        left join idx_sector_reports isr on daily_change.sub_sector = isr.sub_sector
        left join sub_sec_rank ssr on daily_change.sub_sector = ssr.sub_sector
        order by ssr.rank_sub_sec, daily_change.rn;
        """, cur)
    
    ## Top Volume
    top_volume = fetch_query("""
        SELECT symbol, sum(volume) as total_volume
        FROM idx_daily_data
        WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
        AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
        group by symbol
        having max(market_cap) >= 5000000000000
        order by total_volume desc
        limit 5;
        """, cur)
    
    ## Top Value
    top_value = fetch_query("""
        SELECT symbol, sum(volume*close) as total_value
        FROM idx_daily_data
        WHERE date >= DATE_TRUNC('week', CURRENT_DATE)
        AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'
        group by symbol
        having max(market_cap) >= 5000000000000
        order by total_value desc
        limit 5;
        """, cur)
    
    # Page 3 Data

    ## IPO
    df_ipo = fetch_query("""
        select symbol, company_name, offering_end_date, listing_date, offering_price,shares_offered from idx_ipo_details where 
        (listing_date >= DATE_TRUNC('week', CURRENT_DATE)
        AND listing_date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week') or
        current_date between offering_start_date and offering_end_date
        """,cur)
    
    ## IPO Performance
    ipo_perf = fetch_query("""
        select iid.symbol, round((iip.chg_7d*100)::numeric,2) as chg_7d, round((iip.chg_30d*100)::numeric,2) as chg_30d from idx_ipo_details iid 
        join idx_ipo_perf iip on iid.symbol = iip.symbol
        where iid.listing_date between current_date - interval '1 month' and current_date - interval '1 week'
        """, cur)
    
    ## Dividend
    df_div = fetch_query("""
        select iud.symbol, iud.ex_date, iud.payment_date, iud.dividend_amount, round((100 * iud.dividend_amount/idd.close)::numeric,2) as dividend_yield from idx_upcoming_dividend iud  JOIN (
        SELECT DISTINCT ON (symbol)
            symbol, close, date
        FROM idx_daily_data
        ORDER BY symbol, date DESC
        ) idd ON iud.symbol = idd.symbol
        where iud.ex_date > current_date""", cur)

    df_div = df_div.sort_values('ex_date')

    ## Stock split
    stock_split = fetch_query("""
        select symbol, date, split_ratio from idx_stock_split
        where date >= DATE_TRUNC('week', CURRENT_DATE)
        AND date < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '2 week'
        """, cur)
    
    ## Corporate action compilation
    ca_comp = ca_compilation(df_div,df_ipo,stock_split)

    # Generate report
    create_weekly_report(hist_mcap, mcap_changes,top_gainers_losers,indices_changes,sectors_changes,top_3_comp_sectors,top_volume,top_value,df_ipo,stock_split,df_div,ca_comp)

    # Save report as image
    ## Generate the date-based directory path
    date_str = date_generator("cover")
    output_dir = f"pdf_image/{date_str}"
    os.makedirs(output_dir, exist_ok=True)

    ## Convert PDF to list of PIL images (one per page)
    name = f"idx_highlights - {date_generator('cover')}"
    images = convert_from_path(f"pdf_output/{name}.pdf", dpi=300)

    ## Save each page as a PNG in the same directory
    for i, img in enumerate(images):
        img.save(f"{output_dir}/{name}_page_{i+1}.png", "PNG")

    #json
    def df_to_serializable(df):
        """
        Converts datetime columns to string and Decimal columns to float for JSON serialization.
        """
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)
            elif df_copy[col].apply(lambda x: isinstance(x, Decimal)).any():
                df_copy[col] = df_copy[col].astype(float)
            elif df_copy[col].apply(lambda x: isinstance(x, date)).any():
                df_copy[col] = df_copy[col].astype(str)
        return df_copy.to_dict(orient="records")

    # Custom JSON encoder for nested Decimal, Timestamp, and date
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Decimal):
                return float(obj)
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, date):
                return obj.isoformat()
            return super().default(obj)

    # Prepare your compiled dictionary
    compiled_data = {
        "market_cap_history": hist_mcap.replace(np.nan,'null').to_dict(orient="records"),
        "weekly_market_cap_changes": df_to_serializable(mcap_changes)[0] if not mcap_changes.empty else {},
        "top_gainers_losers": df_to_serializable(top_gainers_losers),
        "indices_changes": df_to_serializable(indices_changes),
        "sectors_changes": df_to_serializable(sectors_changes),
        "top_3_companies_per_sector": df_to_serializable(top_3_comp_sectors),
        "top_volume": df_to_serializable(top_volume),
        "top_value": df_to_serializable(top_value),
        "ipo_this_week": df_to_serializable(df_ipo),
        "upcoming_dividends": df_to_serializable(df_div),
        "stock_split_upcoming": df_to_serializable(stock_split),
        "corporate_action_compilation": (
            ca_comp if isinstance(ca_comp, (list, dict)) else df_to_serializable(ca_comp)
        )
    }

    output_folder = "json_output"

    # Scan existing files to find the latest number
    existing_files = [f for f in os.listdir(output_folder) if f.endswith(".json")]

    numbers = []
    for fname in existing_files:
        try:
            num = int(fname.split("_")[0])
            numbers.append(num)
        except (ValueError, IndexError):
            continue  # Skip files that don't follow the pattern

    # Determine next number
    next_number = max(numbers) + 1 if numbers else 27

    # Today's date
    today = datetime.today()
    today_str = (today - timedelta(days=today.weekday())).date().isoformat()

    # Build filename
    output_filename = f"{next_number}_{today_str}.json"

    output_path = os.path.join(output_folder, output_filename)

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(compiled_data, f, indent=4, cls=CustomJSONEncoder)

    print("✅ JSON created")

    # Send Email
    send_email(output_dir)

if __name__ == "__main__":
    main()
