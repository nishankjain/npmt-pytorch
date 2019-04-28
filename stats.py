import math

base_path = './iwslt16.tokenized.de-en/'
# base_path = './orig/de-en/'


def get_stats(file_path):
    num_sentences = 0
    total_num_words = 0
    max_num_words = 0
    max_word_sen = ''
    min_num_words = 100000
    min_word_sen = ''
    vocab = {}
    number_of_words_in_tweets = []
    number_of_word_count_occurrences = {}
    lines_with_multiple_sentences = []

    with open(file_path, 'r') as ip_file:
        for cnt, line in enumerate(ip_file):
            num_sentences += 1
            words = line.split()
            number_of_words = len(words)
            total_num_words += number_of_words
            number_of_words_in_tweets.append(number_of_words)

            if number_of_words < min_num_words:
                min_num_words = number_of_words
                min_word_sen = line
            if number_of_words > max_num_words:
                max_num_words = number_of_words
                max_word_sen = line

            if number_of_words in number_of_word_count_occurrences:
                number_of_word_count_occurrences[number_of_words] += 1
            else:
                number_of_word_count_occurrences[number_of_words] = 1

            num_sentences_in_line = 0
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

                if '.' in word:
                    num_sentences_in_line += 1

            if num_sentences_in_line > 1:
                lines_with_multiple_sentences.append(line)

    # Compute mean
    mean_num_words = total_num_words / num_sentences

    # Compute median
    number_of_words_in_tweets.sort()
    if len(number_of_words_in_tweets) % 2 == 0:
        mid = int(len(number_of_words_in_tweets) / 2)
        median = (number_of_words_in_tweets[mid - 1] + number_of_words_in_tweets[mid]) / 2
    else:
        median = number_of_words_in_tweets[math.floor(len(number_of_words_in_tweets) / 2)]

    # Mode of words
    max_count = 0
    max_count_word = ''
    for (word, count) in vocab.items():
        if count > max_count:
            max_count = count
            max_count_word = word

    # Mode of number of words in tweets
    max_number_of_words_count = 0
    max_number_of_words = 0
    for (number_of_words, count) in number_of_word_count_occurrences.items():
        if count > max_number_of_words_count:
            max_number_of_words_count = count
            max_number_of_words = number_of_words

    # Standard deviation of number of words
    sum_for_std_deviation = 0
    for number_of_words_in_tweet in number_of_words_in_tweets:
        sum_for_std_deviation += ((number_of_words_in_tweet - mean_num_words) ** 2)
    std_deviation = math.sqrt(sum_for_std_deviation / num_sentences)

    print('\n\nFile: {}'.format(file_path))
    print('Total number of sentences: {}'.format(num_sentences))
    print('Total number of words: {}'.format(total_num_words))
    print('Average number of tokens: {}'.format(mean_num_words))
    print('Size of vocabulary: {}'.format(len(vocab)))
    print('Longest sentence ({} words): {}'.format(max_num_words, max_word_sen[:-1]))  # -1 to remove the '\n'
    print('Shortest sentence ({} words): {}'.format(min_num_words, min_word_sen[:-1]))
    print('Median: {}'.format(median))
    print('Mode of words (occurs {} times): {}'.format(max_count, max_count_word))
    print('Mode of number of words ({} sentences): {}'.format(max_number_of_words_count, max_number_of_words))
    print('Standard deviation of number of words: {}'.format(std_deviation))
    print('Number of lines with multiple sentences: {}'.format(len(lines_with_multiple_sentences)))
    # print('Lines with multiple sentences: {}'.format(lines_with_multiple_sentences))


get_stats(base_path + 'valid.de')
get_stats(base_path + 'valid.en')
get_stats(base_path + 'train.de')
get_stats(base_path + 'train.en')
get_stats(base_path + 'test.de')
get_stats(base_path + 'test.en')
