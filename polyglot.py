import re

def percentage_of_bracks(text):
    total_length = len(text)
    text = re.sub(r'[^[]]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_curlies(text):
    total_length = len(text)
    text = re.sub(r'[^{}]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def presence_of_at(text):
    total_length = len(text)
    text = re.sub(r'[^@]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_dollar(text):
    total_length = len(text)
    text = re.sub(r'[^$]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_semi(text):
    total_length = len(text)
    text = re.sub(r'[^;]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_hyphen(text):
    total_length = len(text)
    text = re.sub(r'[^-]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def presence_of_end(text):
    total_length = len(text)
    text = re.findall(r'end', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_def(text):
    total_length = len(text)
    text = re.findall(r'def', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_elif(text):
    total_length = len(text)
    text = re.findall(r'elif', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_return(text):
    total_length = len(text)
    text = re.findall(r'return', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_elsif(text):
    total_length = len(text)
    text = re.findall(r'elsif', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_defun(text):
    total_length = len(text)
    text = re.findall(r'defun', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_object(text):
    total_length = len(text)
    text = re.findall(r'object', text)
    times = len(text)
    return times*5/(total_length)


def presence_of_public(text):
    total_length = len(text)
    text = re.findall(r'public static final', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_func(text):
    total_length = len(text)
    text = re.findall(r'func', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_fun(text):
    total_length = len(text)
    text = re.findall(r'fun', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_static(text):
    total_length = len(text)
    text = re.findall(r'static', text)
    times = len(text)
    return times*5/(total_length)


def presence_of_struct(text):
    total_length = len(text)
    text = re.findall(r'struct', text)
    times = len(text)
    return times*6/(total_length)


def percentage_of_hash(text):
    total_length = len(text)
    text = re.sub(r'[^#]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_percent(text):
    total_length = len(text)
    text = re.sub(r'[^%]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_ast(text):
    total_length = len(text)
    text = re.sub(r'[^*]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_arrow(text):
    total_length = len(text)
    text = re.sub(r'[^<>]', '', text)
    punc_length = len(text)
    return punc_length/total_length

def presence_of_let(text):
    total_length = len(text)
    text = re.findall(r'let', text)
    times = len(text)
    return times*3/(total_length)



def percentage_of_parens(text):
    total_length = len(text)
    text = re.sub(r'[^()]', '', text)
    punc_length = len(text)
    return punc_length/total_length
