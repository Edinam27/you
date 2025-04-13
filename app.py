
import streamlit as st
import re
import os
import time
import json
import random
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# For YouTube scraping and processing
from pytube import YouTube
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
except ImportError:
    # Define dummy classes as placeholder (app will have limited functionality)
    class VideoFileClip:
        def __init__(self, *args, **kwargs): pass
    class AudioFileClip:
        def __init__(self, *args, **kwargs): pass
    class TextClip:
        def __init__(self, *args, **kwargs): pass
    class CompositeVideoClip:
        def __init__(self, *args, **kwargs): pass
    def concatenate_videoclips(*args, **kwargs): pass
    
    print("WARNING: moviepy not available. Video processing features will be disabled.")

# For web scraping and anti-detection
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# For NLP processing and content generation
import nltk
from nltk.tokenize import sent_tokenize
import openai

# Fix the transformers import with proper error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Create a basic dummy pipeline function as fallback
    def pipeline(*args, **kwargs):
        raise NotImplementedError("Transformers library not available or pipeline can't be imported. Please install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers pipeline not available. NLP features will be limited.")


# For image and video processing
from PIL import Image, ImageDraw, ImageFont
import cv2

# For social media posting
import facebook
import instabot
from instagrapi import Client as InstagrapiClient

# For proxy rotation
import requests_random_user_agent
from fp.fp import FreeProxy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ConfigManager:
    """Handles application configuration and environment variables."""
    
    def __init__(self):
        """Initialize configuration manager with default values."""
        self.config = {
            # API keys would typically be in environment variables
            "youtube_api_key": os.getenv("YOUTUBE_API_KEY", ""),
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "facebook_access_token": os.getenv("FACEBOOK_ACCESS_TOKEN", ""),
            "instagram_username": os.getenv("INSTAGRAM_USERNAME", ""),
            "instagram_password": os.getenv("INSTAGRAM_PASSWORD", ""),
            
            # App configuration with defaults
            "download_path": os.getenv("DOWNLOAD_PATH", "./downloads"),
            "output_path": os.getenv("OUTPUT_PATH", "./output"),
            "proxy_list_path": os.getenv("PROXY_LIST_PATH", "./proxies.txt"),
            "user_agent_rotation": True,
            "proxy_rotation": True,
            "delay_min": 3,  # seconds
            "delay_max": 10,  # seconds
            "resize_dims": {
                "instagram_story": (1080, 1920),
                "instagram_post": (1080, 1080),
                "facebook_story": (1080, 1920),
                "facebook_post": (1200, 630),
            },
            "blog_template_path": os.getenv("BLOG_TEMPLATE_PATH", "./templates/blog_template.html"),
        }
        
        # Create necessary directories
        os.makedirs(self.config["download_path"], exist_ok=True)
        os.makedirs(self.config["output_path"], exist_ok=True)
        os.makedirs(os.path.join(self.config["output_path"], "blogs"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_path"], "shorts"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_path"], "posts"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_path"], "audio"), exist_ok=True)
        
    def get_config(self) -> Dict:
        """Return the current configuration."""
        return self.config
    
    def update_config(self, new_config: Dict) -> None:
        """Update the configuration with new values."""
        self.config.update(new_config)
        
    def save_config(self, path: str = "config.json") -> None:
        """Save configuration to a JSON file."""
        # Filter out sensitive information
        safe_config = {k: v for k, v in self.config.items() 
                      if k not in ["youtube_api_key", "openai_api_key", 
                                 "facebook_access_token", "instagram_password"]}
        
        with open(path, 'w') as f:
            json.dump(safe_config, f, indent=4)
    
    def load_config(self, path: str = "config.json") -> None:
        """Load configuration from a JSON file."""
        try:
            with open(path, 'r') as f:
                loaded_config = json.load(f)
                self.update_config(loaded_config)
        except FileNotFoundError:
            logger.warning(f"Config file {path} not found. Using default config.")
        except json.JSONDecodeError:
            logger.error(f"Error parsing config file {path}. Using default config.")


class AntiDetectionManager:
    """Manages anti-bot detection techniques."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration."""
        self.config = config_manager.get_config()
        self.user_agent = UserAgent()
        self.proxies = self._load_proxies()
        self.current_proxy = None
        self.driver = None
        
    def _load_proxies(self) -> List[str]:
        """Load proxies from file or use FreeProxy to get free proxies."""
        proxies = []
        try:
            if os.path.exists(self.config["proxy_list_path"]):
                with open(self.config["proxy_list_path"], "r") as f:
                    proxies = [line.strip() for line in f if line.strip()]
            
            # If no proxies loaded, try to get free proxies
            if not proxies and self.config["proxy_rotation"]:
                for _ in range(5):  # Try to get 5 proxies
                    try:
                        proxy = FreeProxy(rand=True).get()
                        if proxy:
                            proxies.append(proxy)
                    except Exception as e:
                        logger.warning(f"Error getting free proxy: {e}")
                        
            logger.info(f"Loaded {len(proxies)} proxies")
        except Exception as e:
            logger.error(f"Error loading proxies: {e}")
            
        return proxies
    
    def get_random_delay(self) -> float:
        """Get a random delay between min and max."""
        return random.uniform(self.config["delay_min"], self.config["delay_max"])
    
    def rotate_user_agent(self) -> str:
        """Get a random user agent."""
        return self.user_agent.random
    
    def rotate_proxy(self) -> Optional[str]:
        """Get a random proxy from the list."""
        if not self.proxies or not self.config["proxy_rotation"]:
            return None
        
        self.current_proxy = random.choice(self.proxies)
        return self.current_proxy
    
    def get_request_headers(self) -> Dict:
        """Get request headers with a random user agent."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        if self.config["user_agent_rotation"]:
            headers["User-Agent"] = self.rotate_user_agent()
        
        return headers
    
    def get_request_proxies(self) -> Dict:
        """Get request proxies dictionary."""
        if not self.config["proxy_rotation"]:
            return {}
        
        proxy = self.rotate_proxy()
        if not proxy:
            return {}
            
        if proxy.startswith('http'):
            return {
                'http': proxy,
                'https': proxy
            }
        else:
            return {
                'http': f'http://{proxy}',
                'https': f'https://{proxy}'
            }
    
    def initialize_webdriver(self) -> webdriver.Chrome:
        """Initialize and return a webdriver with anti-detection measures."""
        try:
            # Close existing driver if open
            if self.driver:
                self.driver.quit()
            
            options = Options()
            
            # Add undetected-chromedriver specific options
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            
            # Add random user agent if enabled
            if self.config["user_agent_rotation"]:
                options.add_argument(f"user-agent={self.rotate_user_agent()}")
            
            # Add proxy if enabled
            if self.config["proxy_rotation"] and self.current_proxy:
                options.add_argument(f'--proxy-server={self.current_proxy}')
            
            # Create undetected ChromeDriver
            self.driver = uc.Chrome(options=options)
            
            # Execute stealth JS scripts to make automation less detectable
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            # Random additional delay before returning driver
            time.sleep(self.get_random_delay())
            
            return self.driver
            
        except Exception as e:
            logger.error(f"Error initializing webdriver: {e}")
            if self.driver:
                self.driver.quit()
            raise
    
    def close_webdriver(self) -> None:
        """Close the webdriver if it exists."""
        if self.driver:
            self.driver.quit()
            self.driver = None


class YouTubeContentScraper:
    """Handles scraping of YouTube content."""
    
    def __init__(self, config_manager: ConfigManager, anti_detection_manager: AntiDetectionManager):
        """Initialize with configuration and anti-detection manager."""
        self.config = config_manager.get_config()
        self.anti_detection = anti_detection_manager
        self.yt_service = self._initialize_youtube_api() if self.config["youtube_api_key"] else None
        
    def _initialize_youtube_api(self):
        """Initialize the YouTube API client."""
        try:
            api_service_name = "youtube"
            api_version = "v3"
            return googleapiclient.discovery.build(
                api_service_name, api_version, developerKey=self.config["youtube_api_key"])
        except Exception as e:
            logger.error(f"Error initializing YouTube API: {e}")
            return None
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        # Try parsing as URL
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            if '/watch' in parsed_url.path:
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif '/shorts/' in parsed_url.path:
                return parsed_url.path.split('/shorts/')[1]
            elif '/live/' in parsed_url.path:
                # Handle live stream URLs
                live_id = parsed_url.path.split('/live/')[1]
                # Remove any additional path components
                if '/' in live_id:
                    live_id = live_id.split('/')[0]
                return live_id
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path[1:]
        
        # Try direct video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
            
        return None
    
    def get_video_info_api(self, video_id: str) -> Dict:
        """Get video information using YouTube API."""
        if not self.yt_service:
            raise ValueError("YouTube API not initialized. Check API key.")
            
        try:
            request = self.yt_service.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                raise ValueError(f"No video found with ID: {video_id}")
                
            video_data = response['items'][0]
            
            return {
                'id': video_id,
                'title': video_data['snippet']['title'],
                'description': video_data['snippet']['description'],
                'publishedAt': video_data['snippet']['publishedAt'],
                'channelTitle': video_data['snippet']['channelTitle'],
                'channelId': video_data['snippet']['channelId'],
                'thumbnailUrl': video_data['snippet']['thumbnails']['high']['url'],
                'duration': video_data['contentDetails']['duration'],
                'viewCount': video_data['statistics'].get('viewCount', 0),
                'likeCount': video_data['statistics'].get('likeCount', 0),
                'commentCount': video_data['statistics'].get('commentCount', 0),
                'tags': video_data['snippet'].get('tags', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting video info from API: {e}")
            raise
    
    def get_video_info_scraping(self, video_id: str) -> Dict:
        """Get video information using web scraping."""
        driver = None
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            driver = self.anti_detection.initialize_webdriver()
            
            # Add random delay before loading
            time.sleep(self.anti_detection.get_random_delay())
            
            driver.get(url)
            
            # Wait for page to load fully
            time.sleep(self.anti_detection.get_random_delay() * 2)
            
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Extract video title
            title = driver.title.replace(" - YouTube", "")
            
            # Extract description
            description = ""
            description_element = soup.select_one("#description-inline-expander")
            if description_element:
                description = description_element.get_text(strip=True)
            
            # Extract channel info
            channel_element = soup.select_one("#text-container.ytd-channel-name")
            channel_title = channel_element.get_text(strip=True) if channel_element else "Unknown Channel"
            
            # Get view count
            view_count_element = soup.select_one(".view-count")
            view_count = view_count_element.get_text(strip=True) if view_count_element else "0 views"
            view_count = ''.join(filter(str.isdigit, view_count))
            
            # Create response similar to API
            return {
                'id': video_id,
                'title': title,
                'description': description,
                'channelTitle': channel_title,
                'viewCount': view_count,
                'thumbnailUrl': f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                'publishedAt': datetime.now().isoformat(),  # Approximate with current time
                'scraped': True
            }
            
        except Exception as e:
            logger.error(f"Error scraping video info: {e}")
            raise
        finally:
            if driver:
                driver.quit()
    
    def get_video_info(self, video_url: str) -> Dict:
        """Get video information using YouTube API or scraping as fallback."""
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {video_url}")
        
        # Try API first if available
        if self.yt_service:
            try:
                return self.get_video_info_api(video_id)
            except Exception as e:
                logger.warning(f"API method failed, falling back to scraping: {e}")
        
        # Fallback to scraping
        return self.get_video_info_scraping(video_id)
    
    def get_transcript(self, video_id: str) -> List[Dict]:
        """Get video transcript using YouTube Transcript API with improved error handling."""
        try:
            try:
                # Try getting transcript with the default language
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                return transcript_list
            except Exception as e:
                logger.warning(f"Error getting transcript with default language: {e}")
                
                # Try getting available transcript languages
                try:
                    transcript_languages = YouTubeTranscriptApi.list_transcripts(video_id)
                    
                    # Try to find English or any available transcript
                    for transcript in transcript_languages:
                        try:
                            if transcript.language_code == 'en':
                                return transcript.fetch()
                            
                            # Save other language as a fallback
                            other_transcript = transcript
                        except Exception as lang_error:
                            logger.warning(f"Error fetching {transcript.language_code} transcript: {lang_error}")
                    
                    # Use the other language transcript if found
                    if 'other_transcript' in locals():
                        try:
                            # Try translating to English if not already English
                            if other_transcript.language_code != 'en':
                                translated = other_transcript.translate('en')
                                return translated.fetch()
                            else:
                                return other_transcript.fetch()
                        except Exception as trans_error:
                            logger.warning(f"Error translating transcript: {trans_error}")
                            # Return the original transcript as fallback
                            return other_transcript.fetch()
                            
                except Exception as list_error:
                    logger.warning(f"Error listing available transcripts: {list_error}")
                
                # If nothing worked, try Google Speech Recognition or other methods
                # This would require an audio processing pipeline, omitted for brevity
                
                # Return empty list when all methods fail
                logger.error(f"Could not get transcript for video {video_id}")
                return []
                
        except Exception as e:
            logger.error(f"Fatal error getting transcript: {e}")
            return []
    


    # Within the YouTubeContentScraper class:

    def download_video(self, video_url: str, output_path: Optional[str] = None) -> str:
        """Download YouTube video with improved fallback methods."""
        try:
            video_id = self._extract_video_id(video_url)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {video_url}")
            
            if not output_path:
                output_path = os.path.join(self.config["download_path"], f"{video_id}")
                    
            os.makedirs(output_path, exist_ok=True)
            
            # Add delay to avoid detection
            time.sleep(self.anti_detection.get_random_delay())
            
            # Track all errors for detailed reporting
            errors = []
            output_file = os.path.join(output_path, f"{video_id}.mp4")
            
            # Method 1: Try yt-dlp first (most robust)
            try:
                # First make sure yt-dlp is installed
                try:
                    import yt_dlp
                    logger.info("Using yt-dlp for download (recommended method)")
                except ImportError:
                    # If not installed, try to install it
                    logger.warning("yt-dlp not found, attempting to install automatically...")
                    import subprocess
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
                        import yt_dlp
                        logger.info("Successfully installed yt-dlp")
                    except Exception as e:
                        logger.error(f"Could not install yt-dlp: {e}")
                        raise ImportError("yt-dlp could not be installed. Please install manually: pip install yt-dlp")
                
                # Configure yt-dlp with optimal settings
                ydl_opts = {
                    'format': 'best',  # Get best quality
                    'outtmpl': os.path.join(output_path, f"{video_id}.%(ext)s"),
                    'quiet': True,
                    'no_warnings': True,
                    'geo_bypass': True,  # Try to bypass geo-restrictions
                    'cookiefile': os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt") if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")) else None,
                    'nocheckcertificate': True,
                    'ignoreerrors': False
                }
                
                # Add proxy if available
                if self.anti_detection.config["proxy_rotation"] and self.anti_detection.current_proxy:
                    ydl_opts['proxy'] = self.anti_detection.current_proxy
                
                # Add user agent rotation
                if self.anti_detection.config["user_agent_rotation"]:
                    ydl_opts['user_agent'] = self.anti_detection.rotate_user_agent()
                    
                # Download the video
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"Downloading video {video_id} with yt-dlp...")
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                    
                    # Find the downloaded file
                    if info:
                        if 'requested_downloads' in info and info['requested_downloads']:
                            video_path = info['requested_downloads'][0]['filepath']
                        else:
                            video_path = os.path.join(output_path, f"{video_id}.{info.get('ext', 'mp4')}")
                            
                        if os.path.exists(video_path):
                            logger.info(f"Video successfully downloaded to {video_path}")
                            return video_path
                        else:
                            raise FileNotFoundError(f"Expected file not found at {video_path}")
                
            except Exception as e:
                # Don't raise yet, try other methods
                errors.append(f"yt-dlp method failed: {str(e)}")
                logger.warning(f"yt-dlp download failed, trying pytube: {e}")
            
            # Method 2: Try Selenium-based approach
            try:
                logger.info(f"Attempting Selenium-based download for video {video_id}")
                
                # Initialize webdriver with anti-detection measures
                driver = self.anti_detection.initialize_webdriver()
                
                try:
                    # Configure browser for downloads
                    driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
                    params = {
                        'cmd': 'Page.setDownloadBehavior',
                        'params': {
                            'behavior': 'allow',
                            'downloadPath': output_path
                        }
                    }
                    driver.execute("send_command", params)
                    
                    # Navigate to a YouTube downloader service
                    driver.get("https://www.y2mate.com/youtube/" + video_id)
                    time.sleep(self.anti_detection.get_random_delay() * 2)
                    
                    # Click download button (adjust selectors as needed based on the site)
                    try:
                        # Wait for download options to load
                        WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".btn-download"))
                        )
                        
                        # Select MP4 highest quality
                        mp4_buttons = driver.find_elements(By.CSS_SELECTOR, ".btn-download")
                        if mp4_buttons:
                            # Click the first available download button
                            mp4_buttons[0].click()
                            time.sleep(self.anti_detection.get_random_delay() * 3)
                            
                            # Wait for the actual download link to appear and click it
                            download_link = WebDriverWait(driver, 15).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, ".download-link a"))
                            )
                            download_link.click()
                            
                            # Wait for download to complete
                            time.sleep(10)  # Adjust based on average file size
                            
                            # Check if file was downloaded
                            files = os.listdir(output_path)
                            mp4_files = [f for f in files if f.endswith('.mp4') and os.path.getsize(os.path.join(output_path, f)) > 1000000]
                            
                            if mp4_files:
                                video_path = os.path.join(output_path, mp4_files[0])
                                # Rename to consistent filename
                                final_path = os.path.join(output_path, f"{video_id}.mp4")
                                if video_path != final_path:
                                    os.rename(video_path, final_path)
                                logger.info(f"Video downloaded successfully with Selenium to {final_path}")
                                return final_path
                    except Exception as e:
                        logger.error(f"Error during Selenium-based download action: {e}")
                
                finally:
                    # Always close the driver
                    if driver:
                        driver.quit()
                        
            except Exception as e:
                errors.append(f"Selenium method failed: {str(e)}")
                logger.warning(f"Selenium download failed, trying standard pytube: {e}")
            
            # Method 3: Standard pytube (least reliable but try anyway)
            try:
                logger.info(f"Attempting standard pytube download for video {video_id}")
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                
                # Add custom headers to potentially bypass restrictions
                yt.http_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Referer": "https://www.youtube.com/",
                    "Origin": "https://www.youtube.com"
                }
                
                # Try adaptive streams if progressive streams fail
                try:
                    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                    if not video_stream:
                        video_stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
                    
                    if video_stream:
                        video_path = video_stream.download(output_path)
                        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                            logger.info(f"Video downloaded to {video_path} (standard method)")
                            return video_path
                except Exception as pytube_err:
                    # Log the specific pytube error for debugging
                    logger.error(f"Specific pytube download error: {str(pytube_err)}")
                    errors.append(f"Pytube stream error: {str(pytube_err)}")
                    
            except Exception as e:
                errors.append(f"Standard pytube method failed: {str(e)}")
                logger.error(f"All download methods failed for video {video_id}")
            
            # If all methods failed, provide detailed error information
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                error_msg = "Video download failed after multiple attempts. Errors: " + "; ".join(errors)
                
                # Provide a more helpful message about age-restricted content
                if any("age" in err.lower() for err in errors):
                    error_msg += "\nThis video may be age-restricted. Try logging in to YouTube and exporting cookies.txt."
                    
                # Check for geo-restrictions
                if any("geo" in err.lower() or "your country" in err.lower() for err in errors):
                    error_msg += "\nThis video may be geographically restricted. Try using a VPN."
                    
                raise ValueError(error_msg)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise


    
    def download_audio(self, video_url: str, output_path: Optional[str] = None) -> str:
        """Download YouTube video audio with improved error handling."""
        try:
            video_id = self._extract_video_id(video_url)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {video_url}")
            
            if not output_path:
                output_path = os.path.join(self.config["download_path"], f"{video_id}")
                
            os.makedirs(output_path, exist_ok=True)
            
            # Add delay to avoid detection
            time.sleep(self.anti_detection.get_random_delay())
            
            # Track all errors for detailed reporting
            errors = []
            mp3_path = os.path.join(output_path, f"{video_id}.mp3")
            
            # Method 1: Try using yt-dlp first (most reliable)
            try:
                # First check if yt-dlp is available
                try:
                    import yt_dlp
                    logger.info("Using yt-dlp for audio download (recommended method)")
                except ImportError:
                    # If not installed, try to install it
                    logger.warning("yt-dlp not found, attempting to install automatically...")
                    import subprocess
                    import sys
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
                        import yt_dlp
                        logger.info("Successfully installed yt-dlp")
                    except Exception as e:
                        logger.error(f"Could not install yt-dlp: {e}")
                        raise ImportError("Could not install yt-dlp. Please install manually with: pip install yt-dlp")
                
                # Configure yt-dlp for audio extraction
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(output_path, f"{video_id}"),
                    'quiet': True,
                    'no_warnings': True,
                    'geo_bypass': True,
                }
                
                # Add proxy and user agent if configured
                if self.anti_detection.config["proxy_rotation"] and self.anti_detection.current_proxy:
                    ydl_opts['proxy'] = self.anti_detection.current_proxy
                
                if self.anti_detection.config["user_agent_rotation"]:
                    ydl_opts['user_agent'] = self.anti_detection.rotate_user_agent()
                
                # Download audio
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"Downloading audio for video {video_id} with yt-dlp...")
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                
                # Check for the downloaded file
                mp3_file = os.path.join(output_path, f"{video_id}.mp3")
                if os.path.exists(mp3_file) and os.path.getsize(mp3_file) > 0:
                    logger.info(f"Audio downloaded to {mp3_file}")
                    return mp3_file
                    
                # If the above didn't find the file, search for any mp3 in the directory (yt-dlp might have used a different name)
                mp3_files = [f for f in os.listdir(output_path) if f.endswith('.mp3')]
                if mp3_files:
                    found_mp3 = os.path.join(output_path, mp3_files[0])
                    # Rename to our standard format
                    os.rename(found_mp3, mp3_path)
                    logger.info(f"Audio downloaded and renamed to {mp3_path}")
                    return mp3_path
                    
            except Exception as e:
                errors.append(f"yt-dlp audio extraction failed: {str(e)}")
                logger.warning(f"yt-dlp audio extraction failed, trying pytube: {e}")
            
            # Method 2: Try pytube approach
            try:
                logger.info(f"Attempting pytube audio download for video {video_id}")
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                
                # Add custom headers to potentially bypass restrictions
                yt.http_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Referer": "https://www.youtube.com/"
                }
                
                # Try to get audio stream
                audio_stream = yt.streams.filter(only_audio=True).first()
                if not audio_stream:
                    raise ValueError("No audio stream found for this video")
                    
                # Download audio
                audio_path = audio_stream.download(output_path)
                
                # Convert to mp3 if needed
                if audio_path.endswith('.mp4'):
                    try:
                        # Try with moviepy if available
                        audio_file = AudioFileClip(audio_path)
                        mp3_path = audio_path.replace('.mp4', '.mp3')
                        audio_file.write_audiofile(mp3_path)
                        audio_file.close()
                        
                        # Remove the original mp4 audio file
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            
                        logger.info(f"Audio downloaded and converted to {mp3_path}")
                        return mp3_path
                        
                    except Exception as conv_error:
                        # If moviepy fails or isn't available, just return the original format
                        logger.warning(f"Could not convert to mp3: {conv_error}. Keeping original format.")
                        return audio_path
                else:
                    return audio_path
                    
            except Exception as e:
                errors.append(f"pytube audio download failed: {str(e)}")
                logger.error(f"All audio download methods failed for video {video_id}")
            
            # If all methods failed, report the error
            error_msg = "Audio download failed after multiple attempts. Errors: " + "; ".join(errors)
            raise ValueError(error_msg)
                
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise


class ContentProcessor:
    """Processes YouTube content into various formats."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration."""
        self.config = config_manager.get_config()
        
        # Initialize OpenAI if API key is available
        if self.config["openai_api_key"]:
            openai.api_key = self.config["openai_api_key"]
            
        # Initialize text summarizer
        try:
            self.summarizer = pipeline("summarization")
        except Exception as e:
            logger.warning(f"Could not initialize summarizer: {e}")
            self.summarizer = None
    
    def _get_output_path(self, video_id: str, content_type: str) -> str:
        """Get the output path for a specific content type."""
        base_path = os.path.join(self.config["output_path"], content_type)
        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, f"{video_id}")
    
    def create_blog_post(self, video_info: Dict, transcript: List[Dict]) -> str:
        """Create a blog post from video info and transcript."""
        try:
            # Prepare the transcript text
            transcript_text = " ".join([item["text"] for item in transcript])
            
            # Generate blog content with OpenAI if available
            blog_content = ""
            if self.config["openai_api_key"]:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a content writer who creates engaging blog posts from YouTube video transcripts."},
                            {"role": "user", "content": f"Create a well-structured blog post based on this YouTube video titled '{video_info['title']}'. Here's the transcript: {transcript_text[:1000]}... [Transcript continues]. Include an introduction, main points with headings, and a conclusion."}
                        ]
                    )
                    blog_content = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error generating blog with OpenAI: {e}")
            
            # Fallback to basic processing if OpenAI failed or is not available
            if not blog_content:
                # Use transformers summarizer if available
                if self.summarizer:
                    try:
                        # Process in chunks if the text is too long
                        chunks = [transcript_text[i:i+1000] for i in range(0, len(transcript_text), 1000)]
                        summarized_chunks = []
                        
                        for chunk in chunks[:5]:  # Process only first 5 chunks to avoid too long texts
                            summary = self.summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                            summarized_chunks.append(summary[0]['summary_text'])
                        
                        # Combine summaries
                        summary_text = " ".join(summarized_chunks)
                    except Exception as e:
                        logger.error(f"Error summarizing with transformers: {e}")
                        summary_text = transcript_text[:500] + "..."
                else:
                    # Simple extraction of first part
                    summary_text = transcript_text[:500] + "..."
                
                # Create basic blog structure
                sentences = sent_tokenize(transcript_text)
                paragraphs = []
                current_paragraph = []
                
                for i, sentence in enumerate(sentences):
                    current_paragraph.append(sentence)
                    if len(current_paragraph) >= 3 or i == len(sentences) - 1:
                        paragraphs.append(" ".join(current_paragraph))
                        current_paragraph = []
                
                # Build the blog post
                blog_content = f"# {video_info['title']}\n\n"
                blog_content += f"*Published by {video_info.get('channelTitle', 'YouTube Channel')} on {video_info.get('publishedAt', 'Unknown Date')}*\n\n"
                
                # Introduction
                blog_content += f"## Introduction\n\n{summary_text}\n\n"
                
                # Content sections
                blog_content += "## Main Content\n\n"
                for i, paragraph in enumerate(paragraphs[:10]):  # Limit to 10 paragraphs
                    if i % 3 == 0:
                        blog_content += f"### Part {i//3 + 1}\n\n"
                    blog_content += f"{paragraph}\n\n"
                
                # Conclusion
                blog_content += "## Conclusion\n\n"
                blog_content += f"This article was created based on a YouTube video titled '{video_info['title']}'. "
                blog_content += f"For more details, please watch the original video on YouTube.\n\n"
            
            # Save the blog post
            output_dir = self._get_output_path(video_info["id"], "blogs")
            os.makedirs(output_dir, exist_ok=True)
            
            blog_file_path = os.path.join(output_dir, f"{video_info['id']}_blog.md")
            with open(blog_file_path, "w", encoding="utf-8") as f:
                f.write(blog_content)
            
            logger.info(f"Blog post created: {blog_file_path}")
            return blog_file_path
            
        except Exception as e:
            logger.error(f"Error creating blog post: {e}")
            raise
    
    def create_social_media_posts(self, video_info: Dict, transcript: List[Dict]) -> Dict[str, str]:
        """Create social media posts from video info and transcript."""
        try:
            result_paths = {}
            
            # Prepare the transcript text
            transcript_text = " ".join([item["text"] for item in transcript])
            
            # Generate social media content with OpenAI if available
            social_content = {}
            if self.config["openai_api_key"]:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a social media manager who creates engaging posts for different platforms."},
                            {"role": "user", "content": f"Create 3 different social media posts for Instagram and Facebook based on this YouTube video titled '{video_info['title']}'. Here's a summary of the content: {transcript_text[:500]}... [Content continues]. Each post should include hashtags and be engaging. Format your response as JSON with keys 'instagram_1', 'instagram_2', 'instagram_3', 'facebook_1', 'facebook_2', 'facebook_3'."}
                        ]
                    )
                    
                    # Try to parse JSON response
                    content_text = response.choices[0].message.content
                    try:
                        # Extract JSON if wrapped in code blocks
                        if "```json" in content_text:
                            json_str = content_text.split("```json")[1].split("```")[0]
                            social_content = json.loads(json_str)
                        else:
                            social_content = json.loads(content_text)
                    except json.JSONDecodeError:
                        # If not valid JSON, process as text
                        lines = content_text.split("\n")
                        for line in lines:
                            if "instagram_" in line.lower() or "facebook_" in line.lower():
                                parts = line.split(":", 1)
                                if len(parts) == 2:
                                    key = parts[0].strip().lower().replace(" ", "_")
                                    value = parts[1].strip()
                                    social_content[key] = value
                    
                except Exception as e:
                    logger.error(f"Error generating social media content with OpenAI: {e}")
            
            # Fallback to basic processing if OpenAI failed or is not available
            if not social_content:
                # Create basic social media posts
                title_words = video_info['title'].split()
                hashtags = " ".join([f"#{word.lower()}" for word in title_words if len(word) > 3][:5])
                
                social_content = {
                    "instagram_1": f"New post alert! 🚨 Check out this amazing content about {video_info['title']}. {hashtags}",
                    "instagram_2": f"Did you know about {' '.join(title_words[:3])}...? Watch our latest video to learn more! {hashtags}",
                    "instagram_3": f"Content you don't want to miss! 👀 {video_info['title']} {hashtags}",
                    "facebook_1": f"We just uploaded a new video about {video_info['title']}. Click the link in bio to watch the full video!",
                    "facebook_2": f"Interesting facts about {' '.join(title_words[:3])}... Learn more in our latest upload!",
                    "facebook_3": f"Don't miss out on our latest content: {video_info['title']}. Share with someone who needs to see this!"
                }
            
            # Save the social media posts
            output_dir = self._get_output_path(video_info["id"], "posts")
            os.makedirs(output_dir, exist_ok=True)
            
            for platform, content in social_content.items():
                file_path = os.path.join(output_dir, f"{video_info['id']}_{platform}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                result_paths[platform] = file_path
            
            logger.info(f"Social media posts created in {output_dir}")
            return result_paths
            
        except Exception as e:
            logger.error(f"Error creating social media posts: {e}")
            raise
    
 # In the ContentProcessor class

    def create_video_shorts(self, video_path: str, video_info: Dict, transcript: List[Dict]) -> Dict[str, str]:
        """Create short video clips for social media with improved error handling."""
        try:
            result_paths = {}
            
            # Verify video file exists and is valid
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                logger.error(f"Video file not found or is empty: {video_path}")
                raise FileNotFoundError(f"Video file not found or is empty: {video_path}")
            
            # Try to load the video with a timeout and error handling
            try:
                # Load the video
                video = VideoFileClip(video_path)
                
                # Verify video loaded correctly
                if video.duration <= 0 or video.size[0] <= 0 or video.size[1] <= 0:
                    logger.error(f"Video loaded but has invalid properties: duration={video.duration}, size={video.size}")
                    raise ValueError("Video has invalid duration or dimensions")
                    
                logger.info(f"Video loaded successfully: {video_path} (duration: {video.duration}s, size: {video.size})")
            except Exception as video_load_error:
                logger.error(f"Error loading video file: {video_load_error}")
                raise ValueError(f"Could not load video file: {str(video_load_error)}")
            
            # Create output directory
            output_dir = self._get_output_path(video_info["id"], "shorts")
            os.makedirs(output_dir, exist_ok=True)
            
            # Try to find interesting segments based on transcript
            interesting_segments = []
            if transcript:
                # Sort transcript by duration to find segments with most content
                sorted_segments = sorted(transcript, key=lambda x: x['duration'], reverse=True)
                
                # Get top 3 segments
                for segment in sorted_segments[:3]:
                    start_time = segment['start']
                    end_time = start_time + segment['duration']
                    
                    # Extend clip a bit for context
                    start_time = max(0, start_time - 2)
                    end_time = min(video.duration, end_time + 2)
                    
                    if end_time - start_time >= 5:  # Minimum 5 seconds for a short clip
                        interesting_segments.append((start_time, end_time))
            
            # If no interesting segments found based on transcript, create segments based on video duration
            if not interesting_segments:
                total_duration = video.duration
                segment_count = min(3, max(1, int(total_duration / 30)))  # Aim for 30-second clips
                segment_duration = total_duration / segment_count
                
                for i in range(segment_count):
                    start_time = i * segment_duration
                    end_time = min(total_duration, (i + 1) * segment_duration)
                    interesting_segments.append((start_time, end_time))
            
            # Process each interesting segment
            for i, (start_time, end_time) in enumerate(interesting_segments):
                try:
                    logger.info(f"Processing segment {i+1}: {start_time}-{end_time} seconds")
                    
                    # Extract the clip
                    clip = video.subclip(start_time, end_time)
                    
                    # Add title text at the top
                    try:
                        title_text = TextClip(f"{video_info['title'][:50]}{'...' if len(video_info['title']) > 50 else ''}", 
                                        fontsize=24, color='white', 
                                        bg_color='black', size=(clip.w, None), method='caption')
                        title_text = title_text.set_duration(clip.duration)
                        title_text = title_text.set_position(('center', 'top'))
                    except Exception as title_error:
                        logger.warning(f"Error creating title overlay: {title_error}. Using clip without title.")
                        title_text = None
                    
                    # Add source text at the bottom
                    try:
                        source_text = TextClip(f"Source: {video_info.get('channelTitle', 'YouTube')}", 
                                            fontsize=20, color='white', bg_color='black', 
                                            size=(clip.w, None), method='caption')
                        source_text = source_text.set_duration(clip.duration)
                        source_text = source_text.set_position(('center', 'bottom'))
                    except Exception as source_error:
                        logger.warning(f"Error creating source overlay: {source_error}. Using clip without source text.")
                        source_text = None
                    
                    # Combine video with text overlays if available
                    try:
                        clips_to_combine = [clip]
                        if title_text:
                            clips_to_combine.append(title_text)
                        if source_text:
                            clips_to_combine.append(source_text)
                            
                        if len(clips_to_combine) > 1:
                            final_clip = CompositeVideoClip(clips_to_combine)
                        else:
                            final_clip = clip
                    except Exception as composite_error:
                        logger.warning(f"Error creating composite clip: {composite_error}. Using original clip.")
                        final_clip = clip
                    
                    # For Instagram, resize to vertical format if horizontal
                    try:
                        if clip.w > clip.h:
                            # Instagram vertical format
                            try:
                                instagram_clip = final_clip.resize(height=1080)  # Use a smaller size to reduce rendering time
                                # Add black padding on sides
                                instagram_clip = instagram_clip.on_color(
                                    size=(608, 1080),  # 16:9 aspect ratio at 1080p height
                                    color=(0, 0, 0),
                                    pos=('center', 'center')
                                )
                            except Exception as resize_error:
                                logger.warning(f"Error resizing for Instagram: {resize_error}. Using original dimensions.")
                                instagram_clip = final_clip
                            
                            instagram_path = os.path.join(output_dir, f"{video_info['id']}_instagram_short_{i+1}.mp4")
                            instagram_clip.write_videofile(instagram_path, codec='libx264', audio_codec='aac', 
                                                        threads=4, preset='ultrafast')  # Faster encoding
                            result_paths[f'instagram_short_{i+1}'] = instagram_path
                        else:
                            # Already vertical or square
                            instagram_path = os.path.join(output_dir, f"{video_info['id']}_instagram_short_{i+1}.mp4")
                            final_clip.write_videofile(instagram_path, codec='libx264', audio_codec='aac',
                                                    threads=4, preset='ultrafast')
                            result_paths[f'instagram_short_{i+1}'] = instagram_path
                    except Exception as instagram_error:
                        logger.error(f"Error creating Instagram short {i+1}: {instagram_error}")
                    
                    # For Facebook, keep original aspect ratio but ensure within dimensions
                    try:
                        facebook_path = os.path.join(output_dir, f"{video_info['id']}_facebook_short_{i+1}.mp4")
                        final_clip.write_videofile(facebook_path, codec='libx264', audio_codec='aac',
                                                threads=4, preset='ultrafast')
                        result_paths[f'facebook_short_{i+1}'] = facebook_path
                    except Exception as facebook_error:
                        logger.error(f"Error creating Facebook short {i+1}: {facebook_error}")
                    
                except Exception as e:
                    logger.error(f"Error creating short {i+1}: {e}")
            
            # Close the video to free resources
            video.close()
            
            logger.info(f"Created {len(result_paths)} video shorts in {output_dir}")
            
            if not result_paths:
                logger.warning("No video shorts were successfully created")
                
            return result_paths
            
        except Exception as e:
            logger.error(f"Error creating video shorts: {e}", exc_info=True)
            # Return empty dict but don't raise to allow partial success
            return {}
    
    def extract_thumbnail(self, video_path: str, video_info: Dict) -> str:
        """Extract thumbnail from video."""
        try:
            # Create output directory
            output_dir = self._get_output_path(video_info["id"], "thumbnails")
            os.makedirs(output_dir, exist_ok=True)
            
            # Load video and extract frame from middle
            video = VideoFileClip(video_path)
            thumbnail_time = video.duration / 2
            thumbnail = video.get_frame(thumbnail_time)
            
            # Convert to PIL Image
            img = Image.fromarray(thumbnail)
            
            # Resize to standard thumbnail size
            img = img.resize((1280, 720), Image.LANCZOS)
            
            # Add text overlay with title
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                # Use default font if arial not available
                font = ImageFont.load_default()
                
            # Add semi-transparent background for text
            title_text = video_info['title']
            text_width, text_height = draw.textsize(title_text, font=font)
            text_position = ((1280 - text_width) // 2, 720 - text_height - 20)
            
            # Draw background rectangle
            draw.rectangle(
                [text_position[0] - 10, text_position[1] - 10, 
                 text_position[0] + text_width + 10, text_position[1] + text_height + 10],
                fill=(0, 0, 0, 128)
            )
            
            # Draw text
            draw.text(text_position, title_text, font=font, fill=(255, 255, 255))
            
            # Save the thumbnail
            thumbnail_path = os.path.join(output_dir, f"{video_info['id']}_thumbnail.jpg")
            img.save(thumbnail_path, "JPEG")
            
            logger.info(f"Thumbnail extracted to {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error extracting thumbnail: {e}")
            raise


class SocialMediaManager:
    """Manages social media posting and engagement."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration."""
        self.config = config_manager.get_config()
        self.facebook_client = self._initialize_facebook() if self.config["facebook_access_token"] else None
        self.instagram_client = self._initialize_instagram() if (self.config["instagram_username"] and 
                                                                self.config["instagram_password"]) else None
    
    def _initialize_facebook(self):
        """Initialize Facebook API client."""
        try:
            return facebook.GraphAPI(access_token=self.config["facebook_access_token"], version="3.1")
        except Exception as e:
            logger.error(f"Error initializing Facebook API: {e}")
            return None
    
    def _initialize_instagram(self):
        """Initialize Instagram API client."""
        try:
            # Try instagrapi first (more modern and reliable)
            client = InstagrapiClient()
            client.login(self.config["instagram_username"], self.config["instagram_password"])
            return client
        except Exception as e:
            logger.warning(f"Error initializing with instagrapi: {e}, trying instabot...")
            try:
                # Fall back to instabot
                bot = instabot.Bot()
                bot.login(username=self.config["instagram_username"], password=self.config["instagram_password"])
                return bot
            except Exception as e:
                logger.error(f"Error initializing Instagram API: {e}")
                return None
    
    def post_to_facebook(self, content: str, media_path: Optional[str] = None) -> bool:
        """Post content to Facebook."""
        if not self.facebook_client:
            logger.warning("Facebook client not initialized. Check access token.")
            return False
            
        try:
            if media_path:
                # Check if it's a video or image
                if media_path.endswith(('.mp4', '.mov', '.avi')):
                    # Post video
                    with open(media_path, 'rb') as f:
                        self.facebook_client.put_video(f, title=content[:40])
                else:
                    # Post image
                    with open(media_path, 'rb') as f:
                        self.facebook_client.put_photo(f, message=content)
            else:
                # Post text only
                self.facebook_client.put_object(parent_object="me", connection_name="feed", message=content)
                
            logger.info("Successfully posted to Facebook")
            return True
            
        except Exception as e:
            logger.error(f"Error posting to Facebook: {e}")
            return False
    
    def post_to_instagram(self, content: str, media_path: str) -> bool:
        """Post content to Instagram."""
        if not self.instagram_client:
            logger.warning("Instagram client not initialized. Check username and password.")
            return False
            
        try:
            # Check if using instagrapi or instabot
            if isinstance(self.instagram_client, InstagrapiClient):
                # Using instagrapi
                if media_path.endswith(('.mp4', '.mov', '.avi')):
                    # Post video
                    self.instagram_client.video_upload(media_path, caption=content)
                else:
                    # Post image
                    self.instagram_client.photo_upload(media_path, caption=content)
            else:
                # Using instabot
                if media_path.endswith(('.mp4', '.mov', '.avi')):
                    # Post video
                    self.instagram_client.upload_video(media_path, caption=content)
                else:
                    # Post image
                    self.instagram_client.upload_photo(media_path, caption=content)
                
            logger.info("Successfully posted to Instagram")
            return True
            
        except Exception as e:
            logger.error(f"Error posting to Instagram: {e}")
            return False
    
    def schedule_posts(self, platform: str, content_list: List[Tuple[str, Optional[str]]], 
                      start_time: datetime, interval_hours: int = 24) -> List[Dict]:
        """Schedule a series of posts at regular intervals.
        
        Args:
            platform: 'facebook' or 'instagram'
            content_list: List of (content, media_path) tuples
            start_time: When to start posting
            interval_hours: Hours between posts
            
        Returns:
            List of scheduled post details
        """
        schedule = []
        
        for i, (content, media_path) in enumerate(content_list):
            post_time = start_time + pd.Timedelta(hours=i * interval_hours)
            
            schedule.append({
                'platform': platform,
                'content': content,
                'media_path': media_path,
                'scheduled_time': post_time.isoformat(),
                'status': 'scheduled'
            })
        
        # Save schedule to file for later execution by a separate scheduler process
        schedule_dir = os.path.join(self.config["output_path"], "schedules")
        os.makedirs(schedule_dir, exist_ok=True)
        
        schedule_file = os.path.join(schedule_dir, f"schedule_{int(time.time())}.json")
        with open(schedule_file, 'w') as f:
            json.dump(schedule, f, indent=2)
            
        logger.info(f"Created schedule with {len(schedule)} posts in {schedule_file}")
        return schedule
    
    def monitor_engagement(self, platform: str, post_id: str) -> Dict:
        """Monitor engagement metrics for a post."""
        engagement = {
            'likes': 0,
            'comments': 0,
            'shares': 0,
            'views': 0,
            'platform': platform,
            'post_id': post_id,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if platform == 'facebook' and self.facebook_client:
                # Get post insights
                post_data = self.facebook_client.get_object(id=post_id, fields='shares,comments.summary(true),reactions.summary(true)')
                
                engagement['likes'] = post_data.get('reactions', {}).get('summary', {}).get('total_count', 0)
                engagement['comments'] = post_data.get('comments', {}).get('summary', {}).get('total_count', 0)
                engagement['shares'] = post_data.get('shares', {}).get('count', 0)
                
            elif platform == 'instagram' and isinstance(self.instagram_client, InstagrapiClient):
                # Get media info for Instagram if using instagrapi
                media_info = self.instagram_client.media_info(post_id)
                
                engagement['likes'] = media_info.like_count
                engagement['comments'] = media_info.comment_count
                engagement['views'] = getattr(media_info, 'view_count', 0)
            
            logger.info(f"Retrieved engagement metrics for {platform} post {post_id}")
            
        except Exception as e:
            logger.error(f"Error monitoring engagement for {platform} post {post_id}: {e}")
        
        return engagement


class ChannelGrowthManager:
    """Manages channel growth strategies."""
    
    def __init__(self, config_manager: ConfigManager, anti_detection_manager: AntiDetectionManager):
        """Initialize with configuration."""
        self.config = config_manager.get_config()
        self.anti_detection = anti_detection_manager
        
    def find_similar_channels(self, channel_id: str, max_results: int = 5) -> List[Dict]:
        """Find similar channels to the given channel."""
        similar_channels = []
        
        try:
            # Use YouTube API if key available, otherwise scrape
            if self.config["youtube_api_key"]:
                youtube = googleapiclient.discovery.build(
                    "youtube", "v3", developerKey=self.config["youtube_api_key"])
                
                # Get channel information
                channel_response = youtube.channels().list(
                    part="snippet,contentDetails",
                    id=channel_id
                ).execute()
                
                if not channel_response["items"]:
                    logger.warning(f"No channel found with ID: {channel_id}")
                    return similar_channels
                
                # Get channel uploads playlist
                uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
                
                # Get channel's most recent videos
                playlist_response = youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=10
                ).execute()
                
                # Extract video IDs
                video_ids = [item["snippet"]["resourceId"]["videoId"] for item in playlist_response["items"]]
                
                # For each video, find related videos and extract their channel info
                channel_data = {}
                
                for video_id in video_ids[:3]:  # Limit to first 3 videos to avoid quota issues
                    # Get related videos
                    search_response = youtube.search().list(
                        part="snippet",
                        relatedToVideoId=video_id,
                        type="video",
                        maxResults=10
                    ).execute()
                    
                    # Extract channel info from related videos
                    for item in search_response.get("items", []):
                        related_channel_id = item["snippet"]["channelId"]
                        related_channel_title = item["snippet"]["channelTitle"]
                        
                        if related_channel_id != channel_id:  # Skip original channel
                            if related_channel_id not in channel_data:
                                channel_data[related_channel_id] = {
                                    "id": related_channel_id,
                                    "title": related_channel_title,
                                    "count": 1
                                }
                            else:
                                channel_data[related_channel_id]["count"] += 1
                
                # Sort channels by frequency and return top results
                sorted_channels = sorted(channel_data.values(), key=lambda x: x["count"], reverse=True)
                similar_channels = sorted_channels[:max_results]
                
            else:
                # Web scraping approach when API key is not available
                driver = self.anti_detection.initialize_webdriver()
                
                try:
                    # Go to channel page
                    driver.get(f"https://www.youtube.com/channel/{channel_id}/videos")
                    time.sleep(self.anti_detection.get_random_delay())
                    
                    # Get the first few video links
                    video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title")
                    video_urls = [elem.get_attribute("href") for elem in video_elements[:3]]
                    
                    # For each video, find related videos
                    channel_data = {}
                    
                    for video_url in video_urls:
                        if not video_url:
                            continue
                            
                        driver.get(video_url)
                        time.sleep(self.anti_detection.get_random_delay() * 2)
                        
                        # Scroll down to make sure related videos are loaded
                        driver.execute_script("window.scrollBy(0, 500);")
                        time.sleep(self.anti_detection.get_random_delay())
                        
                        # Get related videos' channel information
                        related_elements = driver.find_elements(By.CSS_SELECTOR, "#related ytd-compact-video-renderer")
                        
                        for elem in related_elements[:10]:
                            try:
                                channel_elem = elem.find_element(By.CSS_SELECTOR, ".ytd-channel-name a")
                                related_channel_url = channel_elem.get_attribute("href")
                                related_channel_title = channel_elem.text
                                
                                # Extract channel ID from URL
                                related_channel_id = related_channel_url.split("/")[-1]
                                
                                if related_channel_id != channel_id:  # Skip original channel
                                    if related_channel_id not in channel_data:
                                        channel_data[related_channel_id] = {
                                            "id": related_channel_id,
                                            "title": related_channel_title,
                                            "count": 1
                                        }
                                    else:
                                        channel_data[related_channel_id]["count"] += 1
                            except Exception as e:
                                logger.warning(f"Error extracting channel info from related video: {e}")
                    
                    # Sort channels by frequency and return top results
                    sorted_channels = sorted(channel_data.values(), key=lambda x: x["count"], reverse=True)
                    similar_channels = sorted_channels[:max_results]
                    
                finally:
                    driver.quit()
                    
        except Exception as e:
            logger.error(f"Error finding similar channels: {e}")
        
        return similar_channels
    
    def analyze_trending_keywords(self, category: str = None) -> List[str]:
        """Analyze trending keywords on YouTube."""
        trending_keywords = []
        
        try:
            # Use YouTube API if key available, otherwise scrape
            if self.config["youtube_api_key"]:
                youtube = googleapiclient.discovery.build(
                    "youtube", "v3", developerKey=self.config["youtube_api_key"])
                
                # Get trending videos
                trending_response = youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    chart="mostPopular",
                    regionCode="US",
                    maxResults=50,
                    videoCategoryId=category if category else None
                ).execute()
                
                # Extract titles and descriptions
                text_data = []
                for item in trending_response.get("items", []):
                    text_data.append(item["snippet"]["title"])
                    text_data.append(item["snippet"]["description"])
                    if "tags" in item["snippet"]:
                        text_data.extend(item["snippet"]["tags"])
                
                # Simple keyword extraction
                all_text = " ".join(text_data).lower()
                words = re.findall(r'\b[a-z]{3,15}\b', all_text)
                
                # Remove common stop words
                stop_words = {"the", "and", "you", "that", "have", "for", "this", "with", "not", "are", "from", "your"}
                filtered_words = [word for word in words if word not in stop_words]
                
                # Count word frequency
                word_counts = {}
                for word in filtered_words:
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
                
                # Sort by frequency
                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                trending_keywords = [word for word, count in sorted_words[:20]]
                
            else:
                # Web scraping approach
                driver = self.anti_detection.initialize_webdriver()
                
                try:
                    # Go to trending page
                    driver.get("https://www.youtube.com/feed/trending")
                    time.sleep(self.anti_detection.get_random_delay() * 2)
                    
                    # Get video titles and channel names
                    title_elements = driver.find_elements(By.CSS_SELECTOR, "#video-title")
                    titles = [elem.text for elem in title_elements if elem.text]
                    
                    # Extract keywords
                    all_text = " ".join(titles).lower()
                    words = re.findall(r'\b[a-z]{3,15}\b', all_text)
                    
                    # Remove common stop words
                    stop_words = {"the", "and", "you", "that", "have", "for", "this", "with", "not", "are", "from", "your"}
                    filtered_words = [word for word in words if word not in stop_words]
                    
                    # Count word frequency
                    word_counts = {}
                    for word in filtered_words:
                        if word in word_counts:
                            word_counts[word] += 1
                        else:
                            word_counts[word] = 1
                    
                    # Sort by frequency
                    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                    trending_keywords = [word for word, count in sorted_words[:20]]
                    
                finally:
                    driver.quit()
                    
        except Exception as e:
            logger.error(f"Error analyzing trending keywords: {e}")
        
        return trending_keywords
    
    def suggest_video_ideas(self, channel_id: str, trending_keywords: List[str]) -> List[str]:
        """Suggest video ideas based on channel content and trending keywords."""
        video_ideas = []
        
        try:
            # If OpenAI API is available, use it to generate ideas
            if self.config["openai_api_key"]:
                openai.api_key = self.config["openai_api_key"]
                
                # Get channel info and recent video titles
                channel_info = {}
                recent_titles = []
                
                if self.config["youtube_api_key"]:
                    youtube = googleapiclient.discovery.build(
                        "youtube", "v3", developerKey=self.config["youtube_api_key"])
                    
                    # Get channel information
                    channel_response = youtube.channels().list(
                        part="snippet,contentDetails",
                        id=channel_id
                    ).execute()
                    
                    if channel_response["items"]:
                        channel_info = {
                            "title": channel_response["items"][0]["snippet"]["title"],
                            "description": channel_response["items"][0]["snippet"]["description"]
                        }
                        
                        # Get channel's most recent videos
                        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
                        
                        playlist_response = youtube.playlistItems().list(
                            part="snippet",
                            playlistId=uploads_playlist_id,
                            maxResults=10
                        ).execute()
                        
                        recent_titles = [item["snippet"]["title"] for item in playlist_response["items"]]
                
                # Prepare the prompt for OpenAI
                prompt = f"""
                I need video ideas for a YouTube channel called "{channel_info.get('title', 'my channel')}".
                
                Channel description: {channel_info.get('description', 'No description available')}
                
                Recent video titles:
                {', '.join(recent_titles[:5]) if recent_titles else 'No recent videos available'}
                
                Trending keywords on YouTube: {', '.join(trending_keywords)}
                
                Please suggest 10 engaging video title ideas that would work well for this channel and incorporate some of the trending keywords where appropriate.
                Each title should be compelling and designed to maximize views.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a creative YouTube content strategist."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Parse the response to extract video ideas
                content = response.choices[0].message.content
                lines = content.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('1') or line.startswith('2')):
                        # Clean up formatting
                        idea = re.sub(r'^[\d\-\.\s]+', '', line).strip()
                        if idea:
                            video_ideas.append(idea)
                
            else:
                # Fallback method if no OpenAI API
                # Generate basic ideas combining trending keywords with templates
                templates = [
                    "How to [keyword] in 2023",
                    "Top 10 [keyword] Tips You Need to Know",
                    "Why [keyword] Is Changing Everything",
                    "The Ultimate Guide to [keyword]",
                    "I Tried [keyword] for a Week, Here's What Happened",
                    "[keyword] vs [keyword2]: Which Is Better?",
                    "How [keyword] Is Disrupting [industry]",
                    "[keyword] Mistakes Everyone Makes",
                    "The Truth About [keyword] Nobody Tells You",
                    "Beginners Guide to [keyword]"
                ]
                
                # Generate ideas using templates and keywords
                if trending_keywords:
                    for template in templates:
                        if "[keyword]" in template and "[keyword2]" in template:
                            if len(trending_keywords) >= 2:
                                idea = template.replace("[keyword]", trending_keywords[0])
                                idea = idea.replace("[keyword2]", trending_keywords[1])
                                video_ideas.append(idea)
                        else:
                            for keyword in trending_keywords[:5]:
                                idea = template.replace("[keyword]", keyword)
                                # Replace [industry] with a generic term if present
                                idea = idea.replace("[industry]", "the industry")
                                video_ideas.append(idea)
                
        except Exception as e:
            logger.error(f"Error suggesting video ideas: {e}")
        
        return video_ideas


class YouTubeContentScraperApp:
    """Streamlit application for YouTube content scraping and repurposing."""
    
    def __init__(self):
        """Initialize the application."""
        self.config_manager = ConfigManager()
        self.anti_detection_manager = AntiDetectionManager(self.config_manager)
        self.youtube_scraper = YouTubeContentScraper(self.config_manager, self.anti_detection_manager)
        self.content_processor = ContentProcessor(self.config_manager)
        self.social_media_manager = SocialMediaManager(self.config_manager)
        self.growth_manager = ChannelGrowthManager(self.config_manager, self.anti_detection_manager)
        
        # Set page title and layout
        st.set_page_config(
            page_title="YouTube Content Scraper",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load configuration
        if os.path.exists("config.json"):
            self.config_manager.load_config("config.json")
    
    def run(self):
        """Run the Streamlit application."""
        st.title("YouTube Content Scraper & Repurposer")
        st.sidebar.title("Navigation")
        
        # Navigation options
        page = st.sidebar.radio(
            "Select a function:",
            ["Content Scraping", "Content Repurposing", "Social Media Management", "Channel Growth", "Settings"]
        )
        
        # Display selected page
        if page == "Content Scraping":
            self.content_scraping_page()
        elif page == "Content Repurposing":
            self.content_repurposing_page()
        elif page == "Social Media Management":
            self.social_media_management_page()
        elif page == "Channel Growth":
            self.channel_growth_page()
        elif page == "Settings":
            self.settings_page()

    # In the content_scraping_page method of YouTubeContentScraperApp class

    def content_scraping_page(self):
        """Display the content scraping page with improved error handling."""
        st.header("Content Scraping")
        
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube URL", help="Paste a YouTube video URL here")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            download_video = st.checkbox("Download Video", value=True)
        
        with col2:
            download_audio = st.checkbox("Download Audio", value=True)
        
        with col3:
            get_transcript = st.checkbox("Get Transcript", value=True)
        
        if st.button("Process Video"):
            if youtube_url:
                try:
                    # Extract video info first - this validates the URL
                    with st.spinner("Fetching video information..."):
                        video_info = self.youtube_scraper.get_video_info(youtube_url)
                        
                        # Display video info
                        st.subheader("Video Information")
                        st.write(f"**Title:** {video_info.get('title', 'N/A')}")
                        st.write(f"**Channel:** {video_info.get('channelTitle', 'N/A')}")
                        st.write(f"**Views:** {video_info.get('viewCount', 'N/A')}")
                        st.write(f"**Published:** {video_info.get('publishedAt', 'N/A')}")
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    # Process according to selections
                    video_path = None
                    audio_path = None
                    transcript_data = None
                    
                    # Download video if selected
                    if download_video:
                        with col1:
                            with st.spinner("Downloading video..."):
                                try:
                                    video_path = self.youtube_scraper.download_video(youtube_url)
                                    st.success(f"Video downloaded successfully!")
                                    
                                    # Verify the video file exists and is valid before displaying
                                    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                                        try:
                                            st.video(video_path)
                                        except Exception as video_display_error:
                                            st.warning(f"Video downloaded but cannot be displayed in the browser. You can find it at: {video_path}")
                                            logger.warning(f"Error displaying video: {video_display_error}")
                                    else:
                                        st.warning("Video download completed but the file may be corrupted or empty.")
                                except Exception as video_error:
                                    st.error(f"Error downloading video: {str(video_error)}")
                                    logger.error(f"Video download error: {video_error}", exc_info=True)
                    
                    # Download audio if selected
                    if download_audio:
                        with col2:
                            with st.spinner("Downloading audio..."):
                                try:
                                    audio_path = self.youtube_scraper.download_audio(youtube_url)
                                    st.success(f"Audio downloaded successfully!")
                                    
                                    # Verify audio file exists and is valid before playing
                                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                        try:
                                            st.audio(audio_path)
                                        except Exception as audio_play_error:
                                            st.warning(f"Audio downloaded but cannot be played in the browser. You can find it at: {audio_path}")
                                            logger.warning(f"Error playing audio: {audio_play_error}")
                                    else:
                                        st.warning("Audio download completed but the file may be corrupted or empty.")
                                except Exception as audio_error:
                                    st.error(f"Error downloading audio: {str(audio_error)}")
                                    logger.error(f"Audio download error: {audio_error}", exc_info=True)
                    
                    # Get transcript if selected
                    if get_transcript:
                        with st.spinner("Getting transcript..."):
                            try:
                                video_id = self.youtube_scraper._extract_video_id(youtube_url)
                                transcript_data = self.youtube_scraper.get_transcript(video_id)
                                
                                if transcript_data and len(transcript_data) > 0:
                                    st.subheader("Video Transcript")
                                    
                                    # Display formatted transcript
                                    transcript_text = ""
                                    for entry in transcript_data:
                                        start_time = entry['start']
                                        minutes, seconds = divmod(int(start_time), 60)
                                        transcript_text += f"[{minutes:02d}:{seconds:02d}] {entry['text']}\n"
                                    
                                    st.text_area("Transcript", transcript_text, height=300)
                                    
                                    # Option to save transcript to file
                                    if st.button("Save Transcript to File"):
                                        output_dir = os.path.join(self.config_manager.get_config()["output_path"], "transcripts")
                                        os.makedirs(output_dir, exist_ok=True)
                                        
                                        transcript_file = os.path.join(output_dir, f"{video_id}_transcript.txt")
                                        with open(transcript_file, "w", encoding="utf-8") as f:
                                            f.write(transcript_text)
                                            
                                        st.success(f"Transcript saved to {transcript_file}")
                                else:
                                    st.warning("Could not retrieve transcript for this video. It may not have subtitles or captions.")
                            except Exception as transcript_error:
                                st.warning(f"Error retrieving transcript: {str(transcript_error)}")
                                logger.warning(f"Transcript error: {transcript_error}")
                    
                    # Store metadata for repurposing - even if some steps failed
                    if video_info:
                        try:
                            # Save metadata even if some downloads failed
                            metadata_dir = os.path.join(self.config_manager.get_config()["output_path"], "metadata")
                            os.makedirs(metadata_dir, exist_ok=True)
                            
                            video_id = self.youtube_scraper._extract_video_id(youtube_url)
                            metadata_file = os.path.join(metadata_dir, f"{video_id}_metadata.json")
                            
                            metadata = {
                                "video_info": video_info,
                                "video_path": video_path if video_path and os.path.exists(video_path) else None,
                                "audio_path": audio_path if audio_path and os.path.exists(audio_path) else None,
                                "has_transcript": bool(transcript_data and len(transcript_data) > 0),
                                "processing_date": datetime.now().isoformat()
                            }
                            
                            with open(metadata_file, "w") as f:
                                json.dump(metadata, f, indent=2)
                                
                            st.success("Content processed and metadata saved successfully! You can now repurpose it in the Content Repurposing tab.")
                        except Exception as metadata_error:
                            st.warning(f"Error saving metadata: {str(metadata_error)}")
                            logger.warning(f"Metadata error: {metadata_error}")
                            
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Error in content_scraping_page: {e}", exc_info=True)
            else:
                st.warning("Please enter a YouTube URL")

    def content_repurposing_page(self):
        """Display the content repurposing page."""
        st.header("Content Repurposing")
        
        # Find available content to repurpose
        metadata_dir = os.path.join(self.config_manager.get_config()["output_path"], "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith("_metadata.json")]
        
        if not metadata_files:
            st.info("No content available for repurposing. Please download some videos in the Content Scraping tab first.")
            return
        
        # Create a dropdown to select content
        video_options = {}
        for metadata_file in metadata_files:
            try:
                with open(os.path.join(metadata_dir, metadata_file), "r") as f:
                    metadata = json.load(f)
                    video_title = metadata.get("video_info", {}).get("title", "Unknown")
                    video_id = metadata_file.replace("_metadata.json", "")
                    video_options[f"{video_title} ({video_id})"] = video_id
            except Exception as e:
                logger.error(f"Error loading metadata file {metadata_file}: {e}")
        
        selected_video_option = st.selectbox("Select video to repurpose", list(video_options.keys()))
        video_id = video_options[selected_video_option]
        
        # Load the selected metadata
        with open(os.path.join(metadata_dir, f"{video_id}_metadata.json"), "r") as f:
            metadata = json.load(f)
        
        video_info = metadata["video_info"]
        video_path = metadata["video_path"]
        has_transcript = metadata["has_transcript"]
        
        # Display repurposing options
        st.subheader("Repurposing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_blog = st.checkbox("Create Blog Post", value=True)
        
        with col2:
            create_social = st.checkbox("Create Social Media Posts", value=True)
        
        with col3:
            create_shorts = st.checkbox("Create Video Shorts", value=True)
        
        # Process repurposing
        if st.button("Generate Content"):
            with st.spinner("Repurposing content..."):
                try:
                    # Get transcript if needed and not already loaded
                    transcript_data = []
                    if has_transcript:
                        try:
                            transcript_data = self.youtube_scraper.get_transcript(video_id)
                        except Exception as e:
                            st.warning(f"Could not load transcript: {e}")
                    
                    # Create blog post
                    if create_blog:
                        with st.spinner("Creating blog post..."):
                            try:
                                blog_path = self.content_processor.create_blog_post(video_info, transcript_data)
                                
                                # Read the created blog post and display preview
                                with open(blog_path, "r", encoding="utf-8") as f:
                                    blog_content = f.read()
                                
                                st.subheader("Blog Post Preview")
                                st.markdown(blog_content[:500] + "...")
                                st.success(f"Blog post created successfully at {blog_path}")
                                
                                # Option to view full blog
                                if st.button("View Full Blog Post"):
                                    st.markdown(blog_content)
                                    
                            except Exception as e:
                                st.error(f"Error creating blog post: {str(e)}")
                                logger.error(f"Error creating blog post: {e}", exc_info=True)
                    
                    # Create social media posts
                    if create_social:
                        with st.spinner("Creating social media posts..."):
                            try:
                                social_paths = self.content_processor.create_social_media_posts(video_info, transcript_data)
                                
                                st.subheader("Social Media Posts")
                                
                                # Display social media post previews
                                for platform, path in social_paths.items():
                                    with open(path, "r", encoding="utf-8") as f:
                                        post_content = f.read()
                                    
                                    st.markdown(f"**{platform.replace('_', ' ').title()}**")
                                    st.text_area(f"{platform}", post_content, height=100)
                                
                                st.success(f"Social media posts created successfully!")
                                
                            except Exception as e:
                                st.error(f"Error creating social media posts: {str(e)}")
                                logger.error(f"Error creating social media posts: {e}", exc_info=True)
                    
                    # Create video shorts
                    if create_shorts and video_path:
                        with st.spinner("Creating video shorts (this may take a while)..."):
                            try:
                                shorts_paths = self.content_processor.create_video_shorts(video_path, video_info, transcript_data)
                                
                                st.subheader("Video Shorts")
                                
                                # Display video shorts
                                for platform, path in shorts_paths.items():
                                    st.markdown(f"**{platform.replace('_', ' ').title()}**")
                                    st.video(path)
                                
                                st.success(f"Video shorts created successfully!")
                                
                            except Exception as e:
                                st.error(f"Error creating video shorts: {str(e)}")
                                logger.error(f"Error creating video shorts: {e}", exc_info=True)
                    
                except Exception as e:
                    st.error(f"Error during content repurposing: {str(e)}")
                    logger.error(f"Error in content_repurposing_page: {e}", exc_info=True)
    
    def social_media_management_page(self):
        """Display the social media management page."""
        st.header("Social Media Management")
        
        # Check if API keys are configured
        config = self.config_manager.get_config()
        facebook_configured = bool(config["facebook_access_token"])
        instagram_configured = bool(config["instagram_username"] and config["instagram_password"])
        
        if not facebook_configured and not instagram_configured:
            st.warning("No social media accounts configured. Please add your credentials in the Settings tab.")
            return
        
        # Tabs for different functions
        tab1, tab2, tab3 = st.tabs(["Post Content", "Schedule Posts", "Monitor Engagement"])
        
        with tab1:
            st.subheader("Post to Social Media")
            
            # Platform selection
            platform_options = []
            if facebook_configured:
                platform_options.append("Facebook")
            if instagram_configured:
                platform_options.append("Instagram")
                
            selected_platform = st.selectbox("Select platform", platform_options)
            
            # Content entry
            post_text = st.text_area("Post content", height=150)
            
            # Media upload (required for Instagram)
            media_file = st.file_uploader("Upload media (image or video)", 
                                        type=["jpg", "jpeg", "png", "mp4", "mov"])
            
            # Post button
            if st.button("Post Now"):
                if not post_text and selected_platform == "Facebook":
                    st.error("Please enter some text for your post.")
                elif not media_file and selected_platform == "Instagram":
                    st.error("Instagram requires an image or video.")
                else:
                    with st.spinner(f"Posting to {selected_platform}..."):
                        try:
                            # Save uploaded file if present
                            media_path = None
                            if media_file:
                                media_dir = os.path.join(config["output_path"], "upload_temp")
                                os.makedirs(media_dir, exist_ok=True)
                                
                                file_ext = os.path.splitext(media_file.name)[1]
                                temp_path = os.path.join(media_dir, f"upload_{int(time.time())}{file_ext}")
                                
                                with open(temp_path, "wb") as f:
                                    f.write(media_file.getbuffer())
                                
                                media_path = temp_path
                            
                            # Post to selected platform
                            success = False
                            if selected_platform == "Facebook":
                                success = self.social_media_manager.post_to_facebook(post_text, media_path)
                            elif selected_platform == "Instagram":
                                success = self.social_media_manager.post_to_instagram(post_text, media_path)
                            
                            if success:
                                st.success(f"Posted successfully to {selected_platform}!")
                            else:
                                st.error(f"Failed to post to {selected_platform}. Check logs for details.")
                                
                        except Exception as e:
                            st.error(f"Error posting to {selected_platform}: {str(e)}")
                            logger.error(f"Error posting to {selected_platform}: {e}", exc_info=True)
        
        with tab2:
            st.subheader("Schedule Posts")
            
            # Platform selection
            platform_options = []
            if facebook_configured:
                platform_options.append("Facebook")
            if instagram_configured:
                platform_options.append("Instagram")
                
            selected_platform = st.selectbox("Select platform for scheduling", platform_options)
            
            # Number of posts to schedule
            num_posts = st.number_input("Number of posts to schedule", min_value=1, max_value=10, value=3)
            
            # Start date and time
            start_date = st.date_input("Start date")
            start_time = st.time_input("Start time")
            
            # Interval between posts
            interval_hours = st.number_input("Hours between posts", min_value=1, max_value=168, value=24)
            
            # Create input fields for each post
            st.subheader("Post Content")
            
            post_data = []
            for i in range(num_posts):
                st.markdown(f"**Post {i+1}**")
                post_text = st.text_area(f"Content for post {i+1}", key=f"sched_post_{i}", height=100)
                media_file = st.file_uploader(f"Media for post {i+1}", 
                                            type=["jpg", "jpeg", "png", "mp4", "mov"],
                                            key=f"sched_media_{i}")
                
                post_data.append((post_text, media_file))
                st.markdown("---")
            
            # Schedule button
            if st.button("Schedule Posts"):
                with st.spinner("Scheduling posts..."):
                    try:
                        # Prepare start datetime
                        start_datetime = datetime.combine(start_date, start_time)
                        
                        # Save media files and prepare content list
                        content_list = []
                        for i, (text, media) in enumerate(post_data):
                            media_path = None
                            if media:
                                media_dir = os.path.join(config["output_path"], "scheduled_media")
                                os.makedirs(media_dir, exist_ok=True)
                                
                                file_ext = os.path.splitext(media.name)[1]
                                saved_path = os.path.join(media_dir, f"sched_{int(time.time())}_{i}{file_ext}")
                                
                                with open(saved_path, "wb") as f:
                                    f.write(media.getbuffer())
                                
                                media_path = saved_path
                            
                            if text or media_path:
                                content_list.append((text, media_path))
                        
                        # Schedule the posts
                        schedule = self.social_media_manager.schedule_posts(
                            selected_platform.lower(), content_list, start_datetime, interval_hours)
                        
                        # Display schedule
                        st.subheader("Scheduled Posts")
                        schedule_df = pd.DataFrame(schedule)
                        st.dataframe(schedule_df[['platform', 'content', 'scheduled_time', 'status']])
                        
                        st.success(f"Successfully scheduled {len(schedule)} posts!")
                        
                    except Exception as e:
                        st.error(f"Error scheduling posts: {str(e)}")
                        logger.error(f"Error scheduling posts: {e}", exc_info=True)
        
        with tab3:
            st.subheader("Monitor Engagement")
            
            # Platform selection
            platform_options = []
            if facebook_configured:
                platform_options.append("Facebook")
            if instagram_configured:
                platform_options.append("Instagram")
                
            selected_platform = st.selectbox("Select platform to monitor", platform_options)
            
            # Post ID input
            post_id = st.text_input(f"{selected_platform} post ID", 
                                  help="For Facebook, this is the post ID. For Instagram, this is the media ID.")
            
            # Monitor button
            if st.button("Check Engagement"):
                if not post_id:
                    st.error("Please enter a post ID.")
                else:
                    with st.spinner("Fetching engagement metrics..."):
                        try:
                            engagement = self.social_media_manager.monitor_engagement(
                                selected_platform.lower(), post_id)
                            
                            # Display metrics
                            st.subheader("Engagement Metrics")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Likes", engagement['likes'])
                            
                            with col2:
                                st.metric("Comments", engagement['comments'])
                            
                            with col3:
                                st.metric("Shares", engagement['shares'])
                            
                            with col4:
                                if engagement['views'] > 0:
                                    st.metric("Views", engagement['views'])
                            
                            st.success(f"Successfully retrieved engagement metrics for {selected_platform} post!")
                            
                        except Exception as e:
                            st.error(f"Error monitoring engagement: {str(e)}")
                            logger.error(f"Error monitoring engagement: {e}", exc_info=True)
    
    def channel_growth_page(self):
        """Display the channel growth page."""
        st.header("Channel Growth Manager")
        
        # Tabs for different functions
        tab1, tab2, tab3 = st.tabs(["Find Similar Channels", "Analyze Trends", "Content Ideas"])
        
        with tab1:
            st.subheader("Find Similar Channels")
            
            channel_id = st.text_input("Enter YouTube Channel ID", 
                                     help="You can find this in the channel URL: youtube.com/channel/YOUR_CHANNEL_ID")
            
            max_results = st.slider("Maximum number of results", min_value=3, max_value=20, value=5)
            
            if st.button("Find Similar Channels"):
                if not channel_id:
                    st.error("Please enter a channel ID.")
                else:
                    with st.spinner("Finding similar channels..."):
                        try:
                            similar_channels = self.growth_manager.find_similar_channels(channel_id, max_results)
                            
                            if similar_channels:
                                st.subheader("Similar Channels")
                                
                                for i, channel in enumerate(similar_channels):
                                    st.markdown(f"**{i+1}. {channel['title']}**")
                                    st.markdown(f"Channel ID: `{channel['id']}`")
                                    st.markdown(f"Relevance Score: {channel['count']}")
                                    st.markdown(f"[View Channel](https://www.youtube.com/channel/{channel['id']})")
                                    st.markdown("---")
                                
                                # Save results
                                output_dir = os.path.join(self.config_manager.get_config()["output_path"], "growth")
                                os.makedirs(output_dir, exist_ok=True)
                                
                                results_file = os.path.join(output_dir, f"similar_channels_{channel_id}.json")
                                with open(results_file, "w") as f:
                                    json.dump(similar_channels, f, indent=2)
                                
                                st.success(f"Results saved to {results_file}")
                            else:
                                st.warning("No similar channels found.")
                                
                        except Exception as e:
                            st.error(f"Error finding similar channels: {str(e)}")
                            logger.error(f"Error finding similar channels: {e}", exc_info=True)
        
        with tab2:
            st.subheader("Analyze YouTube Trends")
            
            category_options = {
                "All Categories": None,
                "Film & Animation": "1",
                "Autos & Vehicles": "2",
                "Music": "10",
                "Pets & Animals": "15",
                "Sports": "17",
                "Gaming": "20",
                "People & Blogs": "22",
                "Comedy": "23",
                "Entertainment": "24",
                "News & Politics": "25",
                "Howto & Style": "26",
                "Education": "27",
                "Science & Technology": "28",
                "Nonprofits & Activism": "29"
            }
            
            selected_category = st.selectbox("Select category to analyze", list(category_options.keys()))
            
            if st.button("Analyze Trends"):
                with st.spinner("Analyzing trending content..."):
                    try:
                        category_id = category_options[selected_category]
                        trending_keywords = self.growth_manager.analyze_trending_keywords(category_id)
                        
                        if trending_keywords:
                            st.subheader("Trending Keywords")
                            
                            # Display as word cloud
                            from wordcloud import WordCloud
                            import matplotlib.pyplot as plt
                            
                            # Create word frequency dictionary
                            word_freq = {word: len(word)**1.5 for word in trending_keywords}
                            
                            # Generate word cloud
                            wordcloud = WordCloud(width=800, height=400, 
                                                background_color='white', 
                                                max_words=50).generate_from_frequencies(word_freq)
                            
                            # Display the word cloud
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            st.pyplot(plt)
                            
                            # Display as list
                            st.markdown("### Top Keywords")
                            for i, keyword in enumerate(trending_keywords[:20]):
                                st.markdown(f"{i+1}. **{keyword}**")
                            
                            # Save results
                            output_dir = os.path.join(self.config_manager.get_config()["output_path"], "growth")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            results_file = os.path.join(output_dir, f"trending_keywords_{selected_category.replace(' ', '_')}.json")
                            with open(results_file, "w") as f:
                                json.dump(trending_keywords, f, indent=2)
                            
                            st.success(f"Results saved to {results_file}")
                        else:
                            st.warning("No trending keywords found.")
                            
                    except Exception as e:
                        st.error(f"Error analyzing trends: {str(e)}")
                        logger.error(f"Error analyzing trends: {e}", exc_info=True)
        
        with tab3:
            st.subheader("Generate Video Ideas")
            
            channel_id = st.text_input("Enter YouTube Channel ID for ideas", 
                                     help="You can find this in the channel URL: youtube.com/channel/YOUR_CHANNEL_ID")
            
            # Option to use existing trend analysis
            output_dir = os.path.join(self.config_manager.get_config()["output_path"], "growth")
            trend_files = [f for f in os.listdir(output_dir) if f.startswith("trending_keywords_")] if os.path.exists(output_dir) else []
            
            use_existing = st.checkbox("Use existing trend analysis", value=bool(trend_files))
            
            if use_existing and trend_files:
                selected_file = st.selectbox("Select trend analysis", trend_files)
                with open(os.path.join(output_dir, selected_file), "r") as f:
                    trending_keywords = json.load(f)
            else:
                category_options = {
                    "All Categories": None,
                    "Film & Animation": "1",
                    "Autos & Vehicles": "2",
                    "Music": "10",
                    "Pets & Animals": "15",
                    "Sports": "17",
                    "Gaming": "20",
                    "People & Blogs": "22",
                    "Comedy": "23",
                    "Entertainment": "24",
                    "News & Politics": "25",
                    "Howto & Style": "26",
                    "Education": "27",
                    "Science & Technology": "28",
                    "Nonprofits & Activism": "29"
                }
                
                selected_category = st.selectbox("Select category for trends", list(category_options.keys()))
                trending_keywords = None
            
            if st.button("Generate Video Ideas"):
                if not channel_id:
                    st.error("Please enter a channel ID.")
                else:
                    with st.spinner("Generating video ideas..."):
                        try:
                            # Get trending keywords if not already loaded
                            if not trending_keywords and not use_existing:
                                category_id = category_options[selected_category]
                                trending_keywords = self.growth_manager.analyze_trending_keywords(category_id)
                            
                            if trending_keywords:
                                # Generate video ideas
                                video_ideas = self.growth_manager.suggest_video_ideas(channel_id, trending_keywords)
                                
                                if video_ideas:
                                    st.subheader("Video Ideas")
                                    
                                    for i, idea in enumerate(video_ideas):
                                        st.markdown(f"{i+1}. **{idea}**")
                                    
                                    # Save results
                                    output_dir = os.path.join(self.config_manager.get_config()["output_path"], "growth")
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    results_file = os.path.join(output_dir, f"video_ideas_{channel_id}.json")
                                    with open(results_file, "w") as f:
                                        json.dump(video_ideas, f, indent=2)
                                    
                                    st.success(f"Results saved to {results_file}")
                                else:
                                    st.warning("No video ideas generated.")
                            else:
                                st.warning("No trending keywords found. Cannot generate ideas.")
                                
                        except Exception as e:
                            st.error(f"Error generating video ideas: {str(e)}")
                            logger.error(f"Error generating video ideas: {e}", exc_info=True)
    
    def settings_page(self):
        """Display the settings page."""
        st.header("Settings")
        
        current_config = self.config_manager.get_config()
        
        # Create tabs for different settings
        tab1, tab2, tab3, tab4 = st.tabs(["API Keys", "Paths", "Anti-Detection", "Social Media"])
        
        with tab1:
            st.subheader("API Keys")
            
            youtube_api_key = st.text_input("YouTube API Key", 
                                          value=current_config["youtube_api_key"],
                                          type="password")
            
            openai_api_key = st.text_input("OpenAI API Key", 
                                         value=current_config["openai_api_key"],
                                         type="password")
        
        with tab2:
            st.subheader("File Paths")
            
            download_path = st.text_input("Download Path", 
                                        value=current_config["download_path"])
            
            output_path = st.text_input("Output Path", 
                                      value=current_config["output_path"])
        
        with tab3:
            st.subheader("Anti-Detection Settings")
            
            user_agent_rotation = st.checkbox("Rotate User Agents", 
                                           value=current_config["user_agent_rotation"])
            
            proxy_rotation = st.checkbox("Rotate Proxies", 
                                       value=current_config["proxy_rotation"])
            
            proxy_list_path = st.text_input("Proxy List Path", 
                                          value=current_config["proxy_list_path"])
            
            delay_min = st.number_input("Minimum Delay (seconds)", 
                                      value=current_config["delay_min"],
                                      min_value=1.0,
                                      max_value=60.0)
            
            delay_max = st.number_input("Maximum Delay (seconds)", 
                                      value=current_config["delay_max"],
                                      min_value=delay_min,
                                      max_value=120.0)
        
        with tab4:
            st.subheader("Social Media Settings")
            
            facebook_access_token = st.text_input("Facebook Access Token", 
                                               value=current_config["facebook_access_token"],
                                               type="password")
            
            instagram_username = st.text_input("Instagram Username", 
                                            value=current_config["instagram_username"])
            
            instagram_password = st.text_input("Instagram Password", 
                                            value=current_config["instagram_password"],
                                            type="password")
        
        # Save settings button
        if st.button("Save Settings"):
            try:
                new_config = {
                    "youtube_api_key": youtube_api_key,
                    "openai_api_key": openai_api_key,
                    "download_path": download_path,
                    "output_path": output_path,
                    "user_agent_rotation": user_agent_rotation,
                    "proxy_rotation": proxy_rotation,
                    "proxy_list_path": proxy_list_path,
                    "delay_min": delay_min,
                    "delay_max": delay_max,
                    "facebook_access_token": facebook_access_token,
                    "instagram_username": instagram_username,
                    "instagram_password": instagram_password,
                    "resize_dims": current_config["resize_dims"],  # Keep existing resize dimensions
                    "blog_template_path": current_config["blog_template_path"]  # Keep existing blog template
                }
                
                # Update configuration
                self.config_manager.update_config(new_config)
                self.config_manager.save_config()
                
                # Reinitialize managers with new config
                self.anti_detection_manager = AntiDetectionManager(self.config_manager)
                self.youtube_scraper = YouTubeContentScraper(self.config_manager, self.anti_detection_manager)
                self.content_processor = ContentProcessor(self.config_manager)
                self.social_media_manager = SocialMediaManager(self.config_manager)
                self.growth_manager = ChannelGrowthManager(self.config_manager, self.anti_detection_manager)
                
                st.success("Settings saved successfully!")
                
            except Exception as e:
                st.error(f"Error saving settings: {str(e)}")
                logger.error(f"Error saving settings: {e}", exc_info=True)


# Main entry point
def main():
    """Main entry point for the application."""
    try:
        app = YouTubeContentScraperApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()