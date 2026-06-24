import asyncio
from agent import create_rag_agent, stream_agent_response


async def main():
    agent = await create_rag_agent()

    async for token in stream_agent_response(
        agent,
        "What is SELinux?"
    ):
        print(token, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())