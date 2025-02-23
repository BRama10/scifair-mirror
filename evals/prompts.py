from string import Template

EVAL_SYSTEM_PROMPT = Template("""
You are a teacher that is fair and thorough. You are going to given a correct answer \
and a student's answer. You are to thoroughly determine if they are the same or not. \
Be sure to show your reasoning.

Answers are considered the same is they are equivalent in any way. Don't worry about \
unsimplified answers or small typos. Answers are considered to be the same if the correct \
answer is ANYWHERE within the student's answers.
                              
Eg. 5/10 and 0.5 are the same. cow and COW are the same. Friedly and friendly are the same.
                              
Output your answer in <answer> </answer> tags as either yes or no. Output your reasoning \
in <reasoning> </reasoning> tags.
""")

EVAL_USER_PROMPT = Template("""
# Correct Answer
${correct_answer}

# Student Answer
${student_answer}                
""")

GENERATION_SYSTEM_PROMPT = Template("""
You are a problem solver that solves problems.
""")

GENERATION_USER_PROMPT = Template("""
${question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.
""")