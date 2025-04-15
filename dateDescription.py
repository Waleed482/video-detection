
import time
from datetime import datetime
import sqlite3

def listen_for_date(speech_queue, speech_handler):
    """Listen for the user's spoken date input using SpeechHandler and return the recognized text."""
    try:
        speech_queue.add_speech("Please say the date like '10 April'.")
        
        # Ensure SpeechHandler is listening
        speech_handler.resume_listening()
        
        # Wait for a short period to allow the user to speak
        timeout = 10  # 10 seconds to speak the date
        start_time = time.time()
        spoken_text = None
        
        while time.time() - start_time < timeout:
            spoken_text = speech_handler.get_speech_text()
            # Check if the speech is a valid response (not the default message)
            if spoken_text and spoken_text != "System ready. Waiting for your command..." and "sorry" not in spoken_text.lower():
                speech_handler.pause_listening()  # Pause listening after capturing the date
                return spoken_text
            time.sleep(0.1)  # Small sleep to avoid busy waiting

        # If no valid speech is captured within the timeout
        speech_queue.add_speech("I didn't hear a date. Please try again.")
        speech_handler.pause_listening()
        return None

    except Exception as e:
        speech_queue.add_speech("An unexpected error occurred while listening. Please try again.")
        print(f"Unexpected error in listen_for_date: {e}")
        speech_handler.pause_listening()
        return None

def parse_spoken_date(spoken_text, speech_queue):
    """
    Convert spoken input (e.g., '10 april' or 'tenth april') to 'YYYY-MM-DD' format.
    Assumes the year is 2025 based on existing data.
    """
    if not spoken_text:
        return None
    try:
        # Replace ordinal words with numbers (e.g., 'tenth' -> '10')
        ordinals = {
            'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
            'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
            'eleventh': '11', 'twelfth': '12', 'thirteenth': '13', 'fourteenth': '14',
            'fifteenth': '15', 'sixteenth': '16', 'seventeenth': '17', 'eighteenth': '18',
            'nineteenth': '19', 'twentieth': '20', 'thirtieth': '30', 'thirty first': '31'
        }
        # Also handle 'th', 'rd', 'nd', 'st' suffixes (e.g., '10th' -> '10')
        spoken_text = spoken_text.replace('th ', ' ').replace('rd ', ' ').replace('nd ', ' ').replace('st ', ' ')
        
        # Replace ordinal words with numbers
        for word, number in ordinals.items():
            spoken_text = spoken_text.replace(word, number)

        # Split into day and month (e.g., '10 april' -> day: '10', month: 'april')
        parts = spoken_text.split()
        if len(parts) < 2:
            raise ValueError("Please say the date in 'day month' format, e.g., '10 April' or 'tenth April'.")

        day, month = parts[0], parts[1]
        day = int(day)  # Convert day to integer
        # Convert month name to month number (e.g., 'april' -> 4)
        month = datetime.strptime(month, '%B').month

        # Assume the year is 2025 (based on your data)
        year = 2025

        # Format the date as YYYY-MM-DD (e.g., 2025-04-10)
        formatted_date = f"{year}-{month:02d}-{day:02d}"
        return formatted_date
    except (ValueError, AttributeError) as e:
        speech_queue.add_speech(f"Error parsing the date: {str(e)}")
        print(f"Error parsing date: {e}")
        return None

def fetch_descriptions_for_date(target_date, speech_queue, db_path, conn):
    """
    Fetch descriptions from the database for the given date, print them to the terminal, and speak them.
    Uses the provided db_path and conn for database access.
    """
    cursor = None
    try:
        cursor = conn.cursor()
        # Fetch rows from the descriptions table for the given date
        query = "SELECT * FROM descriptions WHERE date(timestamp) = ? LIMIT 2"
        cursor.execute(query, (target_date,))
        rows = cursor.fetchall()

        # Prepare results
        if rows:
            result_text = f"Descriptions for {target_date}: "
            print(f"\nDescriptions for {target_date}:")
            for row in rows:
                description = row[2]  # The description field
                result_text += f"At {row[1]}, {description}. "
                print(f"At {row[1]}: {description}")
        else:
            result_text = f"No descriptions found for {target_date}."
            print(f"\nNo descriptions found for {target_date}.")

        # Speak the results
        speech_queue.add_speech(result_text)
        return True

    except sqlite3.Error as e:
        error_msg = f"Database error: {str(e)}"
        print(error_msg)
        speech_queue.add_speech(error_msg)
        return False
    finally:
        if cursor:
            cursor.close()