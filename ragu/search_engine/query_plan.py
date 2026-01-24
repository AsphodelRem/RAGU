from typing import List, Dict

from ragu.common.prompts.default_models import SubQuery
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.search_functional import _topological_sort


class QueryPlanEngine(BaseEngine):
    """
    A query planning engine that decomposes complex queries into subqueries.

    It analyzes the input query, breaks it down into a dependency graph of simpler
    subqueries, executes them in topological order, and combines results to produce
    a final answer.

    :param engine: The base search engine used to execute individual subqueries.
    """
    def __init__(self, engine, *args, **kwargs):
        _PROMPTS_NAMES = ["query_decomposition", "query_rewrite"]
        super().__init__(prompts=_PROMPTS_NAMES, *args, **kwargs)
        self.engine: BaseEngine = engine

    async def process_query(self, query: str) -> List[SubQuery]:
        """
        Decompose a complex query into atomic subqueries with dependencies.

        Uses an LLM to analyze the input query and break it down into minimal,
        independent subqueries. Each subquery is assigned a unique ID and may
        declare dependencies on other subqueries that must be resolved first.

        :param query: The complex natural-language query to decompose.
        :return: List of :class:`SubQuery` objects forming a directed acyclic graph (DAG).
        """
        prompt, schema = self.get_prompt("query_decomposition").get_instruction(
            query=query
        )

        response = await self.engine.client.generate(
            prompt=prompt,
            schema=schema
        )
        print(response[0].subqueries)
        return response[0].subqueries

    async def _rewrite_subquery(self, subquery: SubQuery, context: Dict[str, str]) -> str:
        """
        Rewrites a subquery using answers of its dependencies.
        """
        context = {k: v for k, v in context.items() if k in subquery.depends_on}
        prompt, schema = self.get_prompt("query_rewrite").get_instruction(
            original_query=subquery.query,
            context=context
        )
        response = await self.engine.client.generate(
            prompt=prompt,
            schema=schema
        )
        return response[0].query.strip()

    async def _answer_subquery(self, subquery: SubQuery, context: Dict[str, str]) -> str:
        """
        Executes a single subquery.
        Injects answers of dependencies into the prompt.
        """
        if subquery.depends_on:
            query = await self._rewrite_subquery(subquery, context)
        else:
            query = subquery.query

        result = await self.engine.a_query(query)
        return result[0].model_dump().get("response")

    async def a_query(self, query: str) -> str:
        """
        Execute a complex query using the plan-and-execute pipeline.

        This method:
        1. Decompose the query into subqueries with dependencies.
        2. Sort subqueries in topological order.
        3. Rewrite subquery based on previous context and answer the query.
        4. Return the final answer from the last subquery

        Dependent subqueries are automatically rewritten to be self-contained
        by injecting answers from their prerequisite subqueries.

        :param query: The complex natural-language query to answer.
        :return: The final answer as a string.
        """
        subqueries = await self.process_query(query)
        ordered = _topological_sort(subqueries)

        context: Dict[str, str] = {}

        for subquery in ordered:
            answer = await self._answer_subquery(subquery, context)
            context[subquery.id] = answer

        return context[ordered[-1].id]

    async def a_search(self, query, *args, **kwargs):
        """
        Perform a search using the underlying engine.

        :param query: The search query.
        :param args: Additional positional arguments passed to the underlying engine.
        :param kwargs: Additional keyword arguments passed to the underlying engine.
        :return: Search results from the underlying engine.
        """
        return await self.engine.a_search(query, *args, **kwargs)


