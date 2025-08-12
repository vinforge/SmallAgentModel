export interface DistilledProcedure {
  schema_version: "1.0";
  semver: string; // x.y.z
  name: string;
  description?: string;
  intent: {
    signature_text: string;
    embedding_model: string;
    tags?: string[];
    domain?: string;
  };
  scope?: "user" | "team" | "global";
  provenance?: {
    origin?: "auto_distill_success" | "auto_adjustment" | "human_seed";
    author_model?: string;
    notes?: string;
  };
  slots: Array<{
    name: string; // ^[a-zA-Z_][a-zA-Z0-9_]*$
    type: "string" | "number" | "boolean" | "url" | "path" | "datetime" | "enum" | "json";
    enum?: string[];
    description?: string;
    required?: boolean; // default true
  }>;
  preconditions?: Assertion[];
  steps: Step[];
  postconditions?: Assertion[];
  success_criteria: Assertion[];
  tools_required?: string[];
  stats?: {
    times_used?: number;
    success_count?: number;
    failure_count?: number;
    last_used_at?: string; // ISO date
  };
  deprecation?: {
    status?: "active" | "deprecated" | "archived";
    reason?: string;
  };
  x_ext?: Record<string, unknown>;
}

export interface Step {
  id: string; // ^[a-zA-Z0-9_-]+$
  title: string;
  tool: string;
  args: Record<string, unknown>; // may include template placeholders like ${slot}
  condition?: string;
  save_as?: string; // variable name
  expects?: Assertion[];
  retries?: number; // default 0
  on_fail?: {
    policy?: "continue" | "retry" | "fallback" | "abort";
    fallback_to_step_id?: string;
    message?: string;
  };
  notes?: string;
}

export interface Assertion {
  type: "pre" | "post" | "expect" | "success";
  expr: string; // boolean expression
  message?: string;
}

