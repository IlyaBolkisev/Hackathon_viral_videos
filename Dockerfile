FROM 3.11.10-bookworm

ADD ./models ./models
RUN git clone https://github.com/IlyaBolkisev/Hackathon_viral_videos.git
RUN pip install -r requirements.txt

CMD ["python", "main.py"]