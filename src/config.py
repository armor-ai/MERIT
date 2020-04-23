from os import path
import configparser
from configparser import ConfigParser


class Config(object):
    HOME_DIR = path.abspath(path.join(path.dirname(__file__), r'../'))
    __CONFIG_FILE = path.join(HOME_DIR, 'config.ini')
    Parser = ConfigParser()
    Parser.read(__CONFIG_FILE)

    """Section List"""
    __SEC_DATASETS = "DataSets"
    __SEC_VERSIONS = "Versions"
    __SEC_INFO = "Info"
    __SEC_TOPICS = "Topics"
    __SEC_PHRASES = "Phrases"
    __SEC_VALIDATE = "ValidateFiles"
    __SEC_VAL = "Validate"
    __SEC_SENTI = "SentimentLexicon"
    __SEC_MODEL = "Model"
    __SEC_WORDEBD = "WordEmbed"
    __SEC_TopicLable = "TopicLable"

    @classmethod
    def get_section_list(cls):
        return cls.Parser.sections()

    @classmethod
    def __get_attr(cls, attr_type, section, option):
        """
        :param attr_type: can be int, float, str, bool
        :param section:
        :param option:
        :return:
        """
        result = None
        try:
            if attr_type is int:
                result = cls.Parser.getint(section=section, option=option)
            elif attr_type is float:
                result = cls.Parser.getfloat(section=section, option=option)
            elif attr_type is bool:
                result = cls.Parser.getboolean(section=section, option=option)
            elif attr_type is str:
                result = cls.Parser.get(section=section, option=option)
        except configparser.NoOptionError:
            pass
        return result

    @classmethod
    def get_datasets(cls):
        """

        :return: a list of tuples (app, path)
        """
        return cls.Parser.items(cls.__SEC_DATASETS)

    @classmethod
    def get_version_digits(cls):
        return cls.__get_attr(int, cls.__SEC_VERSIONS, "VersionDigits")

    @classmethod
    def get_info_num(cls):
        return cls.__get_attr(int, cls.__SEC_INFO, "InfoNum")

    @classmethod
    def get_topic_num(cls):
        return cls.__get_attr(int, cls.__SEC_TOPICS, "TopicNum")

    @classmethod
    def get_candidate_num(cls):
        return cls.__get_attr(int, cls.__SEC_TOPICS, "CandidateNum")

    @classmethod
    def get_window_size(cls):
        return cls.__get_attr(int, cls.__SEC_TOPICS, "WindowSize")

    @classmethod
    def get_bigram_min(cls):
        return cls.__get_attr(int, cls.__SEC_PHRASES, "Bigram_Min")

    @classmethod
    def get_trigram_min(cls):
        return cls.__get_attr(int, cls.__SEC_PHRASES, "Trigram_Min")

    @classmethod
    def get_senti_lex(cls):
        return cls.__get_attr(str, cls.__SEC_SENTI, "SentiLex")

    @classmethod
    def get_validate_files(cls):
        """

        :return: dictionary of {apk: path_to_changelog}
        """
        return cls.Parser._sections[cls.__SEC_VALIDATE]

    @classmethod
    def get_validate_or_not(cls):
        return cls.__get_attr(int, cls.__SEC_VAL, "Validate")

    @classmethod
    def get_model(cls):
        return cls.__get_attr(str, cls.__SEC_MODEL, "Model")

    @classmethod
    def get_sim_threshold(cls):
        '''
        : get similarity threshold for verification, typically 0.6 or 0.5
        '''
        return cls.__get_attr(float, cls.__SEC_VAL, "SimThreshold")

    @classmethod
    def get_wordembed_or_not(cls):
        '''
        : whether to use the word embeddings for topic labeling
        '''
        return cls.__get_attr(int, cls.__SEC_WORDEBD, "WordEmbed")

    @classmethod
    def get_balance_param(cls):
        '''
        : get balance parameter in float
        '''
        return cls.__get_attr(float, cls.__SEC_TopicLable, "BalanceParam")