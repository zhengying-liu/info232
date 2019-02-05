Instructions for TP2
========
## Table of Contents
* [Step 1: Fetch Materials for TP2 from Upstream](#step-1-fetch-materials-for-tp2-from-upstream)
* [Step 2: Launch Jupyter Notebook and Answer Questions of This TP](#step-2-launch-jupyter-notebook-and-answer-questions-of-this-tp)
* [Step 3: Update Your Own Repo on GitHub Using 'git push' and CHECK](#step-3-update-your-own-repo-on-github-using-git-push-and-check)
* [Step 4: Make a Submission on CodaLab](#step-4-make-a-submission-on-codalab)

## Step 1: Fetch Materials for TP2 from Upstream
Last week, we've forked a GitHub repo from `zhengying-liu/info232` and worked on your own repo copy. From this week now, we will fetch new materials from the original repo and proceed in the same way as last week. To get new materials for TP2, you need to open a terminal and run:
```bash
cd ~/projects/info232
git remote add upstream https://github.com/zhengying-liu/info232.git
git fetch upstream master
git checkout upstream/master TP2
```
and you'll find a new folder `TP2` in your directory, which contains the materials we need for today.

**To Go Further:** what the three command lines above do is 
1. Add the repo `zhengying-liu/info232` as a *remote* named `upstream`. To see the effect of the first command line, you can run `git remote -v` and normally you'll find two remotes: `origin` and `upstream`. The remote `origin` is automatically created when you did `git clone <your-own-GitHub-repo>` last week. And `upstream` is the remote we add this week. For the rest of the course, we'll use this remote to fetch new materials each week;
2. Fetch commits made in the branch `master` of the remote `upstream`. You'll see no explicit changes since the changes are made in the hidden local directory `.git/`, which is the local repo on your computer;
3. Overwrite the (non-existing) directory `TP2/` in the working directory (`~/projects/info232/` on your computer) by replacing with the contents in `upstream/master` (which is in the GitHub repo `zhengying-liu/info232`).

## Step 2: Launch Jupyter Notebook and Answer Questions of This TP
*(This step is the **main part** of this TP, just as last week)*

Normally you should already know how to execute this step without any instructions since it's the almost same as what we did last week.

For those who haven't modified their `~/.bash_profile` file [last time](https://github.com/zhengying-liu/info232/blob/master/TP1/README.md#step-3-launch-jupyter-notebook-and-answer-questions-of-this-tp), you need to first run
```bash
export PATH="/opt/anaconda3/bin:$PATH"
```
to be sure that the command `jupyter-notebook` comes from that of Anaconda 3 (thus for Python 3). 

Then launch the Jupyter notebook of TP2 by running:
```bash
jupyter-notebook --ip=127.0.0.1 ~/projects/info232/TP2/README.ipynb
```
Then you can begin **answering the questions**.

## Step 3: Update Your Own Repo on GitHub Using 'git push' and CHECK

Now you need to update your GitHub repo just as [last week](https://github.com/zhengying-liu/info232/blob/master/TP1/README.md#step-4-update-your-own-repo-on-github-using-git-push), after answering questions in the Jupyter notebook. To do this, you need to:
1. Re-run and save your LATEST notebook by clicking on *Kernel -> Restart and Run all* and then *File -> Save and Checkpoint*;
2. Then run:
	```bash
	cd ~/projects/info232/TP2
	git add *.ipynb *.py
	git commit -m "Si je ne modifie pas ce message de commit, je suis un cochon"
	git push
	```
	(and type your GitHub username and password if necessary);
3. In your favorite browser, go to the page 
	```
	https://github.com/<YOUR-USERNAME>/info232/blob/master/TP2/README.ipynb
	```
	by **using your own username** in the URL.

## Step 4: Make a Submission on CodaLab

After answering the last question of the Jupyter notebook in Step 2, you should now have two new files under the directory `info232/`: `sample_code_submission_<DATE>.zip` and `sample_result_submission_<DATE>.zip`. Either of these 2 files should be a valid submission file for [this Iris competition](https://codalab.lri.fr/competitions/204). Now make a submission and get your new score!

The steps to follow are:
1. Go to the [competition page]((https://codalab.lri.fr/competitions/204)), sign up and accept terms;
2. Under the tag *Participate*, find *Submit / View Results*;
3. Click the button *Submit* and choose one valid submission zip file (could be either code submission or result submission).

If you want to avoid going over the whole Jupyter notebook to generate submission files, the most useful information is in the last question of the notebook. So basically, you need to

- modify sample_code_submission to provide a better model

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).
