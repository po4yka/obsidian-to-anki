from arq.connections import RedisSettings
import inspect

print(inspect.signature(RedisSettings.from_dsn))
print(inspect.signature(RedisSettings.__init__))
