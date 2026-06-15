"""Script to generate a visual Mermaid HTML diagram of the agent's actual execution path for a given Task ID."""
import sys
import sqlite3
import argparse
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, '.')
from ca_agent.graph.graph import graph

def generate_mermaid_html(task_id: str, output_path: str):
    config = {"configurable": {"thread_id": task_id}}
    state = graph.get_state(config)
    
    if not state or not state.values:
        print(f"Error: Task ID '{task_id}' not found in checkpoints database.")
        sys.exit(1)
        
    completed_nodes = state.values.get("completed_nodes", [])
    error_node = state.values.get("error_node")
    
    print(f"Generating trace for Task {task_id}...")
    print(f"Execution path: {' -> '.join(completed_nodes)}")
    
    # Define styling classes for nodes
    mermaid_style = """
    %% Style definitions %%
    classDef unexecuted fill:#eef1f6,stroke:#cbd5e1,stroke-width:1px,color:#94a3b8;
    classDef executed fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#15803d;
    classDef error fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#b91c1c;
    classDef active fill:#fef9c3,stroke:#eab308,stroke-width:2px,color:#a16207;
    """
    
    # Build list of nodes and assign styling classes
    all_nodes = [
        "planner_node", "notification_node", "reconciliation_node",
        "security_master_node", "escalation_gate_node", "action_executor_node", "error_node"
    ]
    
    class_assignments = []
    for node in all_nodes:
        if node == error_node:
            class_assignments.append(f"class {node} error;")
        elif node in completed_nodes:
            class_assignments.append(f"class {node} executed;")
        else:
            class_assignments.append(f"class {node} unexecuted;")
            
    # Add active/pending class if graph is paused on a node
    graph_state = graph.get_state(config)
    if graph_state.next:
        for next_node in graph_state.next:
            class_assignments.append(f"class {next_node} active;")
            
    # Base Mermaid flowchart code matching graph.py structure
    mermaid_code = f"""
flowchart TD
    START([START])
    planner_node[Planner Node]
    notification_node[Notification Node]
    reconciliation_node[Reconciliation Node]
    security_master_node[Security Master Node]
    escalation_gate_node[Escalation Gate Node]
    action_executor_node[Action Executor Node]
    error_node[Error Node]
    END([END])

    START --> planner_node
    
    planner_node --> notification_node
    planner_node --> escalation_gate_node
    planner_node --> error_node
    
    notification_node --> reconciliation_node
    notification_node --> escalation_gate_node
    notification_node --> error_node
    
    reconciliation_node --> security_master_node
    reconciliation_node --> escalation_gate_node
    reconciliation_node --> error_node
    
    security_master_node --> action_executor_node
    security_master_node --> error_node
    
    escalation_gate_node --> action_executor_node
    escalation_gate_node --> error_node
    
    action_executor_node --> END
    error_node --> END

    {mermaid_style}
    {"    ".join(class_assignments)}
    """
    
    # HTML template containing a Mermaid.js live renderer
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Agent Execution Trace - {task_id[:8]}</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f8fafc;
            color: #1e293b;
        }}
        .card {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #0f172a;
            font-size: 24px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #64748b;
            font-size: 14px;
            margin-bottom: 25px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 25px;
            font-size: 13px;
            justify-content: center;
            border-top: 1px solid #e2e8f0;
            padding-top: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .box {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid;
        }}
        .box.executed {{ background-color: #dcfce7; border-color: #22c55e; }}
        .box.active {{ background-color: #fef9c3; border-color: #eab308; }}
        .box.unexecuted {{ background-color: #eef1f6; border-color: #cbd5e1; }}
        .box.error {{ background-color: #fee2e2; border-color: #ef4444; }}
        .mermaid {{
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Corporate Actions Agent Execution Path</h1>
        <div class="subtitle">Task ID: {task_id}</div>
        
        <div class="mermaid">
            {mermaid_code}
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="box executed"></div>
                <span>Executed</span>
            </div>
            <div class="legend-item">
                <div class="box active"></div>
                <span>Paused / Escalated</span>
            </div>
            <div class="legend-item">
                <div class="box error"></div>
                <span>Error / Exception</span>
            </div>
            <div class="legend-item">
                <div class="box unexecuted"></div>
                <span>Unexecuted</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Visual trace HTML file created successfully at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LangGraph Execution Path")
    parser.add_argument("task_id", help="UUID string of the task thread to visualize")
    parser.add_argument("--output", default="execution_trace.html", help="Path to write output HTML file")
    args = parser.parse_args()
    
    generate_mermaid_html(args.task_id, args.output)
