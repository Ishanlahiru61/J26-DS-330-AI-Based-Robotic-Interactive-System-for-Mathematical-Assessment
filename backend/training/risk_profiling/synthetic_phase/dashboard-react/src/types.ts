export type Tier = "High Performer" | "Progressing" | "Needs Urgent Assistance" | "Not enough data";

export interface Skill {
  percentile: number | null;
  level: string | null;
  contribution: number | null;
}

export interface Child {
  id: string;
  grade: number;
  age_months: number;
  class_name: string;
  tier: Tier;
  p_urgent: number | null;
  level: "low" | "monitor" | "moderate" | "high" | "insufficient";
  level_name: string;
  flagged: boolean;
  profile: string | null;
  skills: Record<string, Skill>;
  not_assessed: string[];
  risk_index: number | null;
  practise: string[];
  sessions: { session: number; accuracy: number; median_rt: number | null }[];
  reasons: Record<string, number> | null;
  teacher_report: string[];
  parent_report: string[];
  n_sessions: number;
  n_questions: number;
}

export interface Evidence {
  mastery_validation: { model: string; macro_f1: number; urgent_recall: number }[];
  mastery_selected: string;
  mastery_test: {
    accuracy: number;
    macro_f1: number;
    recall_per_class: Record<string, number>;
    needs_urgent_recall_ci95: [number, number];
    roc_auc_ovr_macro: number;
  };
  mastery_baseline_test: { macro_f1: number; recall_per_class: Record<string, number> };
  confusion_matrix: number[][];
  ablation: { model: string; set: string; macro_f1: number }[];
  risk_models: { model: string; validation_auc: number; real_auc: number }[];
  risk_test: {
    auc: { auc: number; ci95: [number, number] };
    real_auc: { auc: number; ci95: [number, number] };
    screening: { sensitivity: number; specificity: number };
    by_grade: Record<string, number>;
  };
}

export interface Payload {
  meta: { note: string; classes: string[]; skills: string[] };
  children: Child[];
  evidence: Evidence;
}
