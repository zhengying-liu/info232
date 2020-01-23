Instructions for TP1
========
## Table of Contents
* [Step 1: Fetch Materials for TP1 from Upstream](#step-1-fetch-materials-for-tp2-from-upstream)
* [Step 2: Enter the URL of your own Copy of the Jupyter Notebook TP1.ipynb in ChaGrade](#step-2-answer-the-google-forms)
* [Step 3: Launch Jupyter Notebook and Answer Questions of This TP](#step-3-launch-jupyter-notebook-and-answer-questions-of-this-tp)
* [Step 4: Update Your Own Repo on GitHub Using 'git push'](#step-4-update-your-own-repo-on-github-using-git-push)
* [Appendix A: How to Access the Jupyter Notebook from Home](#appendix-a-how-to-access-the-jupyter-notebook-from-home)

## Step 1: Fetch Materials for TP1 from Upstream
Last week, you have forked a GitHub repo from `zhengying-liu/info232` and worked on your own repo copy. From this week now, we will fetch new materials from the original repo and proceed in the same way as last week. To get new materials for TP1, you need to open a terminal and run:
```bash
cd ~/projects/info232
git remote add upstream https://github.com/zhengying-liu/info232.git
git fetch upstream master
git checkout upstream/master TP1
```
and you'll find a new folder `TP1` in your directory, which contains the materials we need for today. If you are curious about what this actually does, see Appendix C.
    
## Step 2: Enter the URL of your own Copy of the Jupyter Notebook TP1.ipynp in ChaGrade
Submit the URL of your Jupyter Notebook in [ChaGrade](https://chagrade.lri.fr/homework/submit/2/27/1/), for homework "1 .  Workflow". This way you make sure that your teacher will know where to find the answers to your TP (otherwise you get 0 points). You then have until January 25 to push new changes.

## Step 3: Launch Jupyter Notebook and Answer Questions of This TP
*(This step is the **main part** of this TP)*

If you use a computer of the university, you can directly go to next paragraph and begin working. But if you choose to use your own computer, mostly you need to first install some softwares such as [Anaconda](https://www.anaconda.com/download/)(with Python 3) and PuTTY for Windows. In this case, turn to the teacher/assistant for help. In the following, we'll suppose you use a Unix-like operating system (Linux, MacOS, etc) or at least a similar shell tool (such as PowerShell for Windows).

In a termimal, run
```
export PATH="/opt/anaconda3/bin:$PATH"
cd ~/projects/info232
jupyter-notebook --ip=127.0.0.1 
```
Then you should see a webpage pop up displaying the directory `~/projects/info232`. Then navigate to `~/projects/info232/TP1` and open the notebook `TP1.ipynb`. (You can also do this by directly running `jupyter-notebook --ip=127.0.0.1 ~/projects/info232/TP1/TP1.ipynb`)

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
git add TP1/TP1.ipynb
git commit -m "Mon second TP est fini!"
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

## Appendix C: Details about Gihub commands
What the three command lines we prescribe do is: 
1. Add the repo `zhengying-liu/info232` as a *remote* named `upstream`. To see the effect of the first command line, you can run `git remote -v` and normally you'll find two remotes: `origin` and `upstream`. The remote `origin` is automatically created when you did `git clone <your-own-GitHub-repo>` last week. And `upstream` is the remote we add this week. For the rest of the course, we'll use this remote to fetch new materials each week;
2. Fetch commits made in the branch `master` of the remote `upstream`. You'll see no explicit changes since the changes are made in the hidden local directory `.git/`, which is the local repo on your computer;
3. Overwrite the (non-existing) directory `TP2/` in the working directory (`~/projects/info232/` on your computer) by replacing with the contents in `upstream/master` (which is in the GitHub repo `zhengying-liu/info232`).

Credits: Zhengying Liu provided these instructions.
