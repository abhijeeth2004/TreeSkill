"""Tool Registry - 用户自定义工具注册系统

这个模块提供了一个灵活的工具注册系统，支持：
1. Python函数工具（直接调用）
2. HTTP API工具（远程调用）
3. MCP工具（Model Context Protocol）
4. 用户自定义工具

使用装饰器或配置文件即可注册，非常用户友好。
"""

from __future__ import annotations

import json
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


# ===========================================================================
# 工具抽象基类
# ===========================================================================

class BaseTool(ABC):
    """所有工具的基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（用于LLM理解）"""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """执行工具"""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """生成OpenAI function calling schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }


# ===========================================================================
# 具体工具实现
# ===========================================================================

@dataclass
class PythonFunctionTool(BaseTool):
    """Python函数工具（直接调用）

    示例：
        @tool("laplace_transform")
        def laplace_transform(expr: str) -> str:
            '''计算拉普拉斯变换'''
            from sympy import laplace_transform, sympify
            expr = sympify(expr)
            result = laplace_transform(expr, t, s)
            return str(result)
    """

    _name: str
    _description: str
    func: Callable
    parameters_schema: Optional[Dict] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, *args, **kwargs) -> Any:
        """直接调用Python函数"""
        logger.info(f"执行工具 [{self.name}]: args={args}, kwargs={kwargs}")
        try:
            result = self.func(*args, **kwargs)
            logger.info(f"工具 [{self.name}] 执行成功: {result}")
            return result
        except Exception as e:
            logger.error(f"工具 [{self.name}] 执行失败: {e}")
            raise

    def to_schema(self) -> Dict[str, Any]:
        """生成schema"""
        if self.parameters_schema:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }

        # 默认schema
        return super().to_schema()


@dataclass
class HTTPTool(BaseTool):
    """HTTP API工具（远程调用）

    示例：
        tool = HTTPTool(
            name="weather_api",
            description="获取天气信息",
            endpoint="https://api.weather.com/current",
            method="GET",
            headers={"Authorization": "Bearer xxx"},
        )

        result = tool.execute(city="Beijing")
    """

    _name: str
    _description: str
    endpoint: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs) -> Any:
        """发送HTTP请求"""
        logger.info(f"执行HTTP工具 [{self.name}]: endpoint={self.endpoint}, params={kwargs}")

        try:
            if self.method.upper() == "GET":
                response = requests.get(
                    self.endpoint,
                    params=kwargs,
                    headers=self.headers,
                    timeout=self.timeout
                )
            else:  # POST
                response = requests.post(
                    self.endpoint,
                    json=kwargs,
                    headers=self.headers,
                    timeout=self.timeout
                )

            response.raise_for_status()
            result = response.json()
            logger.info(f"HTTP工具 [{self.name}] 执行成功")
            return result

        except Exception as e:
            logger.error(f"HTTP工具 [{self.name}] 执行失败: {e}")
            raise


@dataclass
class MCPTool(BaseTool):
    """MCP (Model Context Protocol) 工具

    示例：
        tool = MCPTool(
            name="database_query",
            description="查询数据库",
            mcp_server="localhost:5000",
            tool_name="query",
        )

        result = tool.execute(sql="SELECT * FROM users")
    """

    _name: str
    _description: str
    mcp_server: str
    tool_name: str
    auth_token: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs) -> Any:
        """通过MCP协议调用工具"""
        logger.info(f"执行MCP工具 [{self.name}]: server={self.mcp_server}, params={kwargs}")

        try:
            # MCP协议调用（简化示例）
            payload = {
                "tool": self.tool_name,
                "parameters": kwargs
            }

            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            response = requests.post(
                f"http://{self.mcp_server}/invoke",
                json=payload,
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            logger.info(f"MCP工具 [{self.name}] 执行成功")
            return result

        except Exception as e:
            logger.error(f"MCP工具 [{self.name}] 执行失败: {e}")
            raise


# ===========================================================================
# 工具注册表
# ===========================================================================

class ToolRegistry:
    """工具注册表 - 管理所有注册的工具"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(
        self,
        name: str,
        tool: BaseTool,
        description: Optional[str] = None,
        override: bool = False
    ):
        """注册工具

        参数:
            name: 工具名称
            tool: 工具实例
            description: 工具描述（可选）
            override: 是否覆盖已存在的工具
        """
        if name in self._tools and not override:
            raise ValueError(f"工具 '{name}' 已存在，使用 override=True 来覆盖")

        self._tools[name] = tool
        logger.info(f"✓ 注册工具: {name}")

    def get(self, name: str) -> BaseTool:
        """获取工具"""
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未注册")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())

    def execute(self, name: str, *args, **kwargs) -> Any:
        """执行工具（便捷方法）"""
        tool = self.get(name)
        return tool.execute(*args, **kwargs)

    def load_from_config(self, config_path: Union[str, Path]):
        """从配置文件加载工具

        配置文件格式：
        ```yaml
        tools:
          - name: weather
            type: http
            endpoint: https://api.weather.com/current
            method: GET
            headers:
              Authorization: Bearer xxx

          - name: database
            type: mcp
            mcp_server: localhost:5000
            tool_name: query
        ```
        """
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for tool_config in config.get('tools', []):
            tool = self._create_tool_from_config(tool_config)
            self.register(tool_config['name'], tool)

    def _create_tool_from_config(self, config: Dict) -> BaseTool:
        """根据配置创建工具"""
        tool_type = config.get('type', 'function')

        if tool_type == 'http':
            return HTTPTool(
                _name=config['name'],
                _description=config.get('description', ''),
                endpoint=config['endpoint'],
                method=config.get('method', 'GET'),
                headers=config.get('headers', {}),
            )
        elif tool_type == 'mcp':
            return MCPTool(
                _name=config['name'],
                _description=config.get('description', ''),
                mcp_server=config['mcp_server'],
                tool_name=config.get('tool_name', config['name']),
                auth_token=config.get('auth_token'),
            )
        else:
            raise ValueError(f"不支持的工具类型: {tool_type}")


# 全局工具注册表
tool_registry = ToolRegistry()


# ===========================================================================
# 装饰器 - 用户友好的注册方式
# ===========================================================================

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[Dict] = None,
):
    """装饰器：注册Python函数为工具

    示例1：简单使用
        ```python
        @tool()
        def calculate_laplace(expr: str) -> str:
            '''计算拉普拉斯变换'''
            from sympy import laplace_transform, symbols, sympify
            t, s = symbols('t s')
            expr = sympify(expr)
            result = laplace_transform(expr, t, s)
            return str(result[0])
        ```

    示例2：自定义名称和描述
        ```python
        @tool(name="math_laplace", description="计算数学表达式的拉普拉斯变换")
        def my_laplace_function(expr: str) -> str:
            ...
        ```

    示例3：带schema
        ```python
        @tool(
            name="weather",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        )
        def get_weather(city: str, unit: str = "celsius") -> dict:
            '''获取天气信息'''
            ...
        ```
    """
    def decorator(func: Callable):
        # 工具名称
        tool_name = name or func.__name__

        # 工具描述
        tool_desc = description or func.__doc__ or f"工具: {tool_name}"

        # 创建工具
        tool_instance = PythonFunctionTool(
            _name=tool_name,
            _description=tool_desc,
            func=func,
            parameters_schema=schema,
        )

        # 注册到全局注册表
        tool_registry.register(tool_name, tool_instance)

        # 返回原函数（保持可调用）
        return func

    return decorator


# ===========================================================================
# 便捷函数 - 快速创建工具
# ===========================================================================

def create_http_tool(
    name: str,
    endpoint: str,
    description: str = "",
    method: str = "GET",
    headers: Optional[Dict] = None,
) -> HTTPTool:
    """快速创建HTTP工具

    示例：
        ```python
        weather_tool = create_http_tool(
            name="weather",
            endpoint="https://api.weather.com/current",
            description="获取天气信息",
            headers={"Authorization": "Bearer xxx"}
        )

        result = weather_tool.execute(city="Beijing")
        ```
    """
    tool = HTTPTool(
        _name=name,
        _description=description,
        endpoint=endpoint,
        method=method,
        headers=headers or {},
    )

    tool_registry.register(name, tool)
    return tool


def create_mcp_tool(
    name: str,
    mcp_server: str,
    tool_name: str,
    description: str = "",
    auth_token: Optional[str] = None,
) -> MCPTool:
    """快速创建MCP工具

    示例：
        ```python
        db_tool = create_mcp_tool(
            name="database",
            mcp_server="localhost:5000",
            tool_name="query",
            description="查询数据库"
        )

        result = db_tool.execute(sql="SELECT * FROM users")
        ```
    """
    tool = MCPTool(
        _name=name,
        _description=description,
        mcp_server=mcp_server,
        tool_name=tool_name,
        auth_token=auth_token,
    )

    tool_registry.register(name, tool)
    return tool


# ===========================================================================
# 导出
# ===========================================================================

__all__ = [
    # 基类
    "BaseTool",

    # 工具类型
    "PythonFunctionTool",
    "HTTPTool",
    "MCPTool",

    # 注册表
    "ToolRegistry",
    "tool_registry",

    # 装饰器
    "tool",

    # 便捷函数
    "create_http_tool",
    "create_mcp_tool",
]
