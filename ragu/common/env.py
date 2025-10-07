from pydantic_settings import BaseSettings


class Env(BaseSettings):
    """
    A configuration class that loads settings from environment variables using Pydantic.

    This class defines the settings required to connect to an LLM API, such as the base URL
    and API key. These values are loaded from environment variables, as specified in the
    `.env` file.

    :param llm_base_url: The base URL for the LLM API.
    :param llm_api_key: The API key used for authenticating requests to the LLM API.
    :return: An instance of the `Settings` class with values loaded from the environment.
    """
    llm_model_name: str
    llm_base_url: str
    llm_api_key: str

    class Config:
        """
        Configuration class to specify the location of environment variables.

        :param env_file: The file where environment variables are loaded from (e.g., `.env`).
        """
        env_file = ".env"

    @classmethod
    def from_env(cls, env_path: str | None):
        """
        Creates an instance of `Settings`, loading environment variables from a specified file.

        :param env_path: Path to the `.env` file (optional). If None, uses the default `.env`.
        :return: An instance of `Settings` with values loaded from the specified file.
        """
        return cls(_env_file=env_path) if env_path else cls()
