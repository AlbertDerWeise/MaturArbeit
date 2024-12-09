# #🄽euro-optimized 🄶eneration from 🅁eddit  -   🄽🄶🅁                                  
__________________________

## About

**NGR** is a Reddit post classification tool designed to classify posts using artificial intelligence as well as deterministic algorithms. The tool analyzes Reddit posts and assigns tags based on various categories which are to be denoted as the following ones: `school`, `games`, `work`, `family`, `everyday_life/health`, `money`, `relationships`, `animals`, `sports`, `science/technology`, `clothing`, `mental_state`, `literature/television`, `nostalgia`, `drugs`, `celebrities`, `society_and_social_procedures`, `gender_specific`, `common_relatability`, `politics`, `sex`, `music/art`, `ethnicity/culture/languages`, `political_incorrectness`, and `misc`.

Additionally, the tool optimizes viewership based on given feedback (such as likes, views, comments on videos) by adjusting and optimizing the tags present in future videos. Currently, it is fresh out of testing stage, leaving the code unaesthetic, patchworked, and purely functional. If you must use this tool, consider taking the time to making it sightly. 
____
## Viewership Optimization

**NGR** employs a feedback loop mechanism to enhance viewership of TikTok videos generated from Reddit posts. After compiling and publishing TikTok videos, the tool gathers feedback metrics such as likes, views, and comments. Based on this feedback, the tool analyzes the performance of the video and identifies trends in audience engagement.

### Feedback Analysis

So far, **NGR** has only been tested synthetically, meaning its feedback is generated by a function. Should you be up for the challenge, try to implement it in a real-world scenario and see what happens.

### Optimization Strategy

- **Tag Adjustment**: Based on the feedback analysis, the tool adjusts the tags assigned to future Reddit posts. It identifies tags that correlate with higher engagement metrics and prioritizes them in subsequent video compilations.
- **Content Enhancement**: The tool may recommend content enhancements or modifications based on feedback to better resonate with the audience.
- **Posting Schedule**: By analyzing the timing of engagement, the tool suggests optimal posting schedules to maximize viewership.

### Continuous Improvement

**NGR** iteratively refines its optimization strategies based on ongoing feedback and performance analysis. By continuously adapting to audience preferences and trends, the tool ensures consistent improvement in viewership and engagement metrics.
____
## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/AlbertDerWeise/MaturArbeit.git
    ```

2. **Install dependencies**

    - Navigate to the project directory
    
    ```bash
    cd MaturArbeit
    ```
    
    - Install the required Python packages using pip
    
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API credentials**

    - **Reddit API Credentials**:
        - Obtain your Reddit API credentials from the Reddit Developer Console.
        - Create a file named `credentials.pkl` in the project directory.
        - Store your Reddit API credentials in the `credentials.pkl` file in the following format:

        ```python
        {
            'id': 'your_client_id',
            'secret': 'your_client_secret',
            'username': 'your_reddit_username',
            'pwd': 'your_reddit_password',
            'openai': 'your_openai_api_key',
            'prompt': 'your_openai_prompt',
            'prompt2': 'your_second_openai_prompt',
        }
        ```

        Replace `'your_client_id'`, `'your_client_secret'`, `'your_reddit_username'`, `'your_reddit_password'`, `'your_openai_api_key'`, `'your_openai_prompt'`, and `'your_second_openai_prompt'` with your actual Reddit API credentials, OpenAI API key, and LLM prompts


4. **Setup Tesseract OCR (if necessary)**

    - If you are using Windows, ensure that the Tesseract OCR executable path is correctly set. You can set it using the following command:

    ```python
    pyt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```

    Replace the path with the actual path to your Tesseract OCR executable.

5. **Prepare assets**
   - Either download the `src/assets/` folder, or make one yourself - keep in mind: the background video _needs to be_ in `mp4` format and the background music files _need to be_ in `wav` format. Furthermore, it is advised for the background video to be shortened to the ***desired maximum video length*** as videos of excess length may cause unnecessary delay in rendering. The same is applicable to background songs, too.
   - Make sure a `src/assets/` directory is present and located as specified. **Otherwise, the program will not load properly.**
   - make sure a `src/assets/models` directory is present prepared and set up with at least 1 directory such that coqui is able to access `.wav` files for finetuning purposes. Prepared models are to be downloaded from the link in the description as they are too large to fit into a git repo. Obviously, you may use any array of `.wav` files you desire.

____
## Usage
To use **NGR** to classify Reddit posts and compile them into a TikTok video, follow these steps:



1. **Configure API Credentials**: Ensure you have set up the necessary API credentials for Reddit, OpenAI, and Eleven Labs as described in the installation instructions.

2. **Adjust Parameters (Optional)**: If desired, you can adjust parameters such as the number of Reddit posts to fetch, the tags to prioritize, and any other settings within the code.

3. **Uncomment the folowing line at the start of the assemble script:** ``` #fetch_data.CommentFetcher.__init__(self=10, time='month', subreddit='meirl') ``` 

4. **Run the Tool**: Execute the main script to start the classification process. This script fetches Reddit posts, analyzes them using AI to assign tags, and compiles them into a TikTok video by running:
 ```bash
python3 assemble.py
```

5. **Review Output**: Once the script completes execution, review the output files generated. These may include the compiled TikTok video, text files containing tag data, and any other relevant logs or analytics.
Optimize Tags (Optional): Based on the feedback received from viewership metrics (likes, views, comments), you may choose to optimize the tags used for future Reddit posts. This optimization process helps enhance engagement and viewership for future TikTok videos.
Iterate and Improve: Continuously iterate on the classification process, incorporating feedback and making adjustments as necessary to improve the quality and performance of the generated TikTok videos.
By following these steps, you can effectively utilize NGR to classify Reddit posts and create engaging TikTok content tailored to your audience's interests and preferences. 


6. **Integrate the algorithm into the assembling function**: Inspect the algorithm in ```train.py``` and integrate it into ```assemble.py```


7. **Create a script to post at given intervals:** Create a script that runs ```assemble.py``` (as well as data retrieval) at given intervals and posts the results subsequently


8. **Tweak scripts:** Tweak the scripts to suit your needs, change the LLM, access Reddit stories instead of memes, decide for yourself. The scripts are written legibly to facilitate tweaking.
____

## File Overview:

- ```assemble.py```: Assembles a video from ground up by calling the files responsible for individual steps


- ```fetch_data.py```: Retrieves data from Reddit and saves it to a local database


- ```compile_vid.py```: Takes video resources and images or voice as input and returns a video


- ```indexer.py```: Responsible for database structure


- ```plotter.py```: Interesting plots regarding algorithm testing statistics


- ```tts.py```: Finetunes a coqui model based on input voice samples


- ```splitwav.py```: Splits input voice lines into small chunks for ```tts.py```


- ```tesseract_showcase.py```: Self-explanatory


- ```testdumpling.py```: Is used to regenerate the entire database, be careful


- ```train.py```: Includes a testing-stage implementation of the tag adjustment algorithm and has yet to be implemented into the entire process


- ```ttstest.py```: Self-explanatory
