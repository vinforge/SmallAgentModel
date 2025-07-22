graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "test_node_1"
    labels "TestNode"
    labels "Concept"
    name "Test Concept"
    description "A test concept for Phase A validation"
    created_at "2025-06-17T10:22:31.848466"
  ]
  node [
    id 1
    label "test_node_2"
    labels "TestNode"
    labels "Memory"
    content "Test memory content"
    created_at "2025-06-17T10:22:31.848510"
  ]
  edge [
    source 0
    target 1
    key "test_rel_1"
    type "RELATES_TO"
    strength 0.8
    created_at "2025-06-17T10:22:31.848515"
  ]
]
