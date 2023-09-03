from abc import ABC, abstractmethod

class EnvCreator(ABC):

    @abstractmethod
    def get_env_name():
        pass

    @abstractmethod
    def create_env():
        pass

    @abstractmethod
    def get_centrilazied():
        pass
    
    @abstractmethod
    def get_decentrilazied():
        pass
    
