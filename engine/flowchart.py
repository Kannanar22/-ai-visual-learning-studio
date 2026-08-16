"""
Converts Python source code into a Mermaid.js flowchart definition using
only the built-in `ast` module. Fully offline, no AI calls.
"""
import ast


class _FlowchartBuilder:
    def __init__(self):
        self.lines = ["flowchart TD"]
        self.counter = 0
        self.edges = []

    def _new_id(self) -> str:
        self.counter += 1
        return f"N{self.counter}"

    def _label(self, node_id: str, text: str, shape: str = "box"):
        text = text.replace('"', "'")
        if shape == "decision":
            self.lines.append(f'    {node_id}{{"{text}"}}')
        elif shape == "start_end":
            self.lines.append(f'    {node_id}(["{text}"])')
        else:
            self.lines.append(f'    {node_id}["{text}"]')

    def _edge(self, a: str, b: str, label: str = ""):
        if label:
            self.lines.append(f'    {a} -- {label} --> {b}')
        else:
            self.lines.append(f'    {a} --> {b}')

    def _expr_to_text(self, node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "..."

    def build_block(self, body, entry_id) -> str:
        """Process a list of statements sequentially; returns the exit node id."""
        current = entry_id
        for stmt in body:
            current = self.build_stmt(stmt, current)
        return current

    def build_stmt(self, stmt, prev_id) -> str:
        if isinstance(stmt, ast.If):
            return self._build_if(stmt, prev_id)

        elif isinstance(stmt, ast.For):
            loop_id = self._new_id()
            self._label(loop_id, f"For {self._expr_to_text(stmt.target)} in "
                                  f"{self._expr_to_text(stmt.iter)}", "decision")
            self._edge(prev_id, loop_id)
            body_exit = self.build_block(stmt.body, loop_id)
            self._edge(body_exit, loop_id, "loop")
            after_id = self._new_id()
            self._label(after_id, "After loop")
            self._edge(loop_id, after_id, "done")
            return after_id

        elif isinstance(stmt, ast.While):
            loop_id = self._new_id()
            self._label(loop_id, f"While {self._expr_to_text(stmt.test)}", "decision")
            self._edge(prev_id, loop_id)
            body_exit = self.build_block(stmt.body, loop_id)
            self._edge(body_exit, loop_id, "loop")
            after_id = self._new_id()
            self._label(after_id, "After loop")
            self._edge(loop_id, after_id, "done")
            return after_id

        elif isinstance(stmt, ast.Return):
            ret_id = self._new_id()
            val = self._expr_to_text(stmt.value) if stmt.value else ""
            self._label(ret_id, f"Return {val}", "start_end")
            self._edge(prev_id, ret_id)
            return ret_id

        else:
            step_id = self._new_id()
            text = self._expr_to_text(stmt)
            self._label(step_id, text)
            self._edge(prev_id, step_id)
            return step_id

    def _build_if(self, stmt: ast.If, prev_id: str) -> str:
        cond_id = self._new_id()
        self._label(cond_id, self._expr_to_text(stmt.test) + "?", "decision")
        self._edge(prev_id, cond_id)

        merge_id = self._new_id()
        self._label(merge_id, "Continue")

        # Yes branch
        if stmt.body:
            yes_entry = self._new_id()
            self._label(yes_entry, self._expr_to_text(stmt.body[0]))
            self._edge(cond_id, yes_entry, "Yes")
            yes_exit = yes_entry
            for extra in stmt.body[1:]:
                yes_exit = self.build_stmt(extra, yes_exit)
            self._edge(yes_exit, merge_id)
        else:
            self._edge(cond_id, merge_id, "Yes")

        # No / else branch
        if stmt.orelse:
            no_entry = self._new_id()
            self._label(no_entry, self._expr_to_text(stmt.orelse[0]))
            self._edge(cond_id, no_entry, "No")
            no_exit = no_entry
            for extra in stmt.orelse[1:]:
                no_exit = self.build_stmt(extra, no_exit)
            self._edge(no_exit, merge_id)
        else:
            self._edge(cond_id, merge_id, "No")

        return merge_id

    def render(self) -> str:
        return "\n".join(self.lines)


def generate_flowchart(code: str) -> str:
    """
    Parse Python source and produce a Mermaid `flowchart TD` definition.
    Returns the Mermaid source as a string.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f'flowchart TD\n    A["Syntax Error: {str(e.msg).replace(chr(34), chr(39))}"]'

    builder = _FlowchartBuilder()
    start_id = builder._new_id()
    builder._label(start_id, "Start", "start_end")

    exit_id = builder.build_block(tree.body, start_id)

    end_id = builder._new_id()
    builder._label(end_id, "End", "start_end")
    builder._edge(exit_id, end_id)

    return builder.render()
