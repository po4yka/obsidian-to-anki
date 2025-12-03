from arq.connections import RedisSettings

settings = RedisSettings.from_dsn("redis://localhost:6379")
print(f"Original conn_timeout: {settings.conn_timeout}")
try:
    settings.conn_timeout = 10
    print(f"Modified conn_timeout: {settings.conn_timeout}")
except Exception as e:
    print(f"Error modifying: {e}")

print(f"Has socket_timeout? {hasattr(settings, 'socket_timeout')}")
