Instructions for TP1
========
## Table of Contents
* [Step 1: Create a Copy of This Repo using <em>Fork</em>](#step-1-create-a-copy-of-this-repo-using-fork)
* [Step 2: Answer the Google Forms](#step-2-answer-the-google-forms)
* [Step 3: Launch Jupyter Notebook and Answer Questions of This TP](#step-3-launch-jupyter-notebook-and-answer-questions-of-this-tp)
* [Step 4: Update Your Own Repo on GitHub Using 'git push'](#step-4-update-your-own-repo-on-github-using-git-push)
* [Appendix: How to Access the Jupyter Notebook from Home](#appendix-how-to-access-the-jupyter-notebook-from-home)

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

    For more information on *Cloning a repository*, check [this guide page](https://help.github.com/articles/cloning-a-repository/). From next TP, we will need to sync with the original repo, i.e. `zhengying-liu/info232`, and you can check [this guide](https://help.github.com/articles/fork-a-repo/#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository) in advance.
    
## Step 2: Answer the Google Forms
Submit the name of your repo in [this Google Forms](https://docs.google.com/forms/d/e/1FAIpQLScNHMlgRwoKqvVJGGhF-WJtpcxAxnPq_gYYLnJM2TmmaYLQhw/viewform?usp=sf_link). This way you make sure that your teacher will know where to find the answers to your TP (otherwise you get 0 points). You then have until February 3 to push new changes.

## Step 3: Launch Jupyter Notebook and Answer Questions of This TP
*(This step is the **main part** of this TP)*

If you use a computer of the university, you can directly go to next paragraph and begin working. But if you choose to use your own computer, mostly you need to first install some softwares such as [Anaconda](https://www.anaconda.com/download/)(with Python 3) and PuTTY for Windows. In this case, turn to the teacher/assistant for help. In the following, we'll suppose you use a Unix-like operating system (Linux, MacOS, etc) or at least a similar shell tool (such as PowerShell for Windows).

In a termimal, run
```
export PATH="/opt/anaconda3/bin:$PATH"
cd ~/projects/info232
jupyter-notebook --ip=127.0.0.1 
```
Then you should see a webpage pop up displaying the directory `~/projects/info232`. Then navigate to `~/projects/info232/TP1` and open the notebook `README.ipynb`. (You can also do this by directly running `jupyter-notebook --ip=127.0.0.1 ~/projects/info232/TP1/README.ipynb`)

**Then answer the questions in this notebook. You don't have to answer all questions to have full score (5/5). Only 5 correct answers will do.**

In this whole course, we'll use Python 3 instead of Python 2. But by default, the system will use Python 2 when you use the command `python` directly in terminal (type `which python` to see why). This explains why we ran the line 
`export PATH="/opt/anaconda3/bin:$PATH"`. It prioritizes the search of the command `python` (or `jupyter`, `jupyter-notebook`, etc) in the Anaconda3 directory `/opt/anaconda3/bin` first. And if you want to avoid running this command everytime when you launch a terminal, you can append it to the file `~/.bash_profile` by following command line:
```
echo 'export PATH="/opt/anaconda3/bin:$PATH"' >> ~/.bash_profile
```
Then it'll be automatically run when launching a terminal.

## Step 4: Update Your Own Repo on GitHub Using 'git push'
After answering all questions in the Jupyter notebook in this TP, you need to update these changes to your remote repo on GitHub, such that the teacher can look at it and give you a score!

To do this, first make sure all the cells of your notebook are run and your LATEST notebook is saved:
* Use  Kernel + Restart and Run all.
* Save with File + Save and Checkpoint.

Then only you can open a terminal and run
```bash
cd ~/projects/info232
git add TP1/README.ipynb
git commit -m "Mon premier TP a ete juste parfait!"
git push
```
Probably you'll be asked to type your GitHub username and password. Make sure to use the same username as the one that you used to fork the repo.

*Git is a very useful and powerful tool for version controling and building open source software. For a tutorial, [here](http://rogerdudler.github.io/git-guide/) is a simple one.*

**CE N'EST PAS FINI!!** 
Go to your repo and verify that your answers are in there!
EVERY TIME YOU "commit", USE A DIFFERENT MESSAGE IN -m "message" to track your changes.

## Appendix A: How to Access the Jupyter Notebook from Home
It could happen that you don't manage to finish the TP in class. But fortunately, you can continue working on it even when you are at home. The steps to follow are:
1. On a local machine, run: 
    ```bash
    ssh PRENOM.NOM@tp-ssh1.dep-informatique.u-psud.fr
    ```
    (use your OWN name!) and type your password. Now it's like we are on a remote machine;
2. On the remote machine, run: 
    ```bash
    jupyter-notebook --ip=127.0.0.1 --no-browser --port=8889
    ```
    Make sure that the command `jupyter-notebook` is that of Anaconda 3, see [Step 3](#step-3-launch-jupyter-notebook-and-answer-questions-of-this-tp). Then you will see a URL which begins by  `http://127.0.0.1:8889/?token=4785b6...`. Copy the token.
    
3. On the local machine, run: 
    ```bash
    ssh -N -f -L localhost:8888:localhost:8889 PRENOM.NOM@tp-ssh1.dep-informatique.u-psud.fr
    ```
    (again use your own name and password)
4. Then on your local machine, you can remotely access the Jupyter notebook using the URL `http://localhost:8888` with your favorite browser. If this is the first time you access this Jupyter notebook, you need to provide a token, which you already copied previously.

## Appendix B: How to run your TP on your own Windows machine
* Download and install [git](https://git-scm.com/download/win) . You will get a console MINGW64 at teh same time, use it to process all the commands provided above.
* Instead of these instructions
    export PATH="/opt/anaconda3/bin:$PATH"
    cd ~/projects/info232
    jupyter-notebook --ip=127.0.0.1 
 just click on the Jupyter icon in the menu.

Credits: Isabelle Shao provided these instructions.
