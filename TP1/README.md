Instructions for TP1
========

## Step 1: Create a Copy of This Repo using *Fork*
To finish this TP, you first need to create a copy of this GitHub repository under your own GitHub account. We will use a feature that GitHub provides called *forking*. To do this, you need to

1. **Create your own GitHub account** (if you don't already have one): click on **Sign Up** on this page (if you are not already signed in) then follow the instructions. 

2. **Fork the repo**: sign in with your GitHub account then click on the **Fork** button on the top-right of this page. You will then have a copy of this GitHub repo under your own account (but not yet locally on your computer since everything is still on the server of GitHub remotely). 

    For more information on *forking*, you can check [this guide page](https://help.github.com/articles/fork-a-repo/);
    
3. **Create local copy of the repo**: on the webpage of your own repo (NOT the original one at `zhengying-liu/info232`!), click the green button **Clone or download** and copy the URL to clipboard (the URL should be something like `https://github.com/YOUR-USERNAME/info232.git`). Then, open a terminal (Google it if you don't know what it means) on your own computer and type

    ```bash
    cd ~
    mkdir projects
    cd projects/
    git clone https://github.com/YOUR-USERNAME/info232.git
    ```
    
    Above command lines create a directory `projects/` in your home directory (`~`) and make a local copy (by `git clone`) using the URL you just copied. Now you should have a directory at `~/projects/info232` with all materials we need for this TP.

    For more info on *Cloning a repository*, check [this guide page](https://help.github.com/articles/cloning-a-repository/). From next TP, we will need to sync with the original repo, i.e. `zhengying-liu/info232`, and you can check [this guide](https://help.github.com/articles/fork-a-repo/#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository) in advance.

## Step 2: Setup Docker and Launch Jupyter Notebook 
*(This step is the **main part** of this TP)*

To make sure that all participants in data challenges all share the same computing resources and have access to necessary Python libraries, we use [Docker](https://opensource.com/resources/what-docker) to provide virtual environments (called Docker containers) that are shared by all participants. So we'll also use Docker in our TP.

If you work with a computer of the university, you should already have Docker installed. If you use your own personal computer, google 'how to install Docker'. After setting up Docker, run
```
docker run -it -p 8888:8888 codalab/codalab-legacy:py3
```
Then in the container, launch Jupyter notebook by running
```
jupyter3-notebook --ip=127.0.0.1
```
In the popped page in Jupyter notebook, navigate to `~/projects/info232/TP1` and open the notebook `README.ipynb`

**Then answer the questions in this notebook. You don't have to answer all questions to have full score (5/5). Only 5 correct answers will do.**

In the whole course following, we'll use Python 3 instead of Python 2. But by default, the system will use Python 2 when you use the command `python` in terminal (type `which python` to see why). So in each session of terminal, you can prioritize Python 3 by running
```
export PATH="/opt/anaconda3/bin:$PATH"
```
And if you want to run this line automatically at each time you launch a terminal, you can append it to the file `~/.bash_profile` by following command line:
```
echo 'export PATH="/opt/anaconda3/bin:$PATH"' >> ~/.bash_profile
```

**This step is to be improved**

## Step 3: Update Your Own Repo on GitHub Using 'git push'
After answering all questions in the Jupyter notebook in this TP, you need to update these changes to your remote repo on GitHub, such that the teacher can look at it and give you a score!

To do this, you can open a terminal and run
```bash
cd ~/projects/info232
git add TP1/README.ipynb
git commit -m "Mon premier TP a ete juste parfait!"
git push
```
Probably you'll be asked to type your GitHub username and password. Make sure to use the same username as the one that you used to fork the repo.

## Step 4: Answer the Google Forms
At last, please fill in [this Google Forms](https://docs.google.com/forms/d/e/1FAIpQLScNHMlgRwoKqvVJGGhF-WJtpcxAxnPq_gYYLnJM2TmmaYLQhw/viewform?usp=sf_link).
