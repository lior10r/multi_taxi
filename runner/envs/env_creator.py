from abc import ABC, abstractmethod

class EnvCreator(ABC):

    @abstractmethod
    def get_env_name():
        pass

    @abstractmethod
    def create_env():
        pass

    @abstractmethod
    def get_centralized():
        pass
    
    @abstractmethod
    def get_decentralized():
        pass
    
