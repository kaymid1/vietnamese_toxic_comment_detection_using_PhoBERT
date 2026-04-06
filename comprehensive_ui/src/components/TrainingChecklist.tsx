import { useState, type DragEvent } from "react";
import { GripVertical, Plus, Trash2 } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../app/components/ui/accordion";
import { Badge } from "../app/components/ui/badge";
import { Card } from "../app/components/ui/card";
import { Checkbox } from "../app/components/ui/checkbox";
import { Button } from "../app/components/ui/button";
import { type PhaseStatus, type TrainingTaskDefinition, useTrainingStore } from "../hooks/useTrainingStore";

const STATUS_TEXT: Record<PhaseStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  done: "Done",
};

const STATUS_CLASS: Record<PhaseStatus, string> = {
  not_started: "bg-muted text-muted-foreground",
  in_progress: "bg-background-warning text-text-warning",
  done: "bg-background-success text-text-success",
};

const renderTaskLabel = (task: TrainingTaskDefinition) => {
  if (!task.param) {
    return <span>{task.label}</span>;
  }

  return (
    <span>
      {task.label} — param:{" "}
      <code className="rounded bg-violet-500/10 px-1.5 py-0.5 font-mono text-xs text-cyan-600 dark:text-cyan-300">
        {task.param}
      </code>
    </span>
  );
};

interface EditState {
  mode: "phase" | "group" | "task";
  id: string;
  phaseId: string;
  groupId: string | null;
  label: string;
  param: string;
}

type DragItem =
  | { kind: "phase"; phaseId: string }
  | { kind: "group"; phaseId: string; groupId: string }
  | { kind: "task"; phaseId: string; groupId: string | null; taskId: string };

const reorderIds = (ids: string[], draggedId: string, targetId: string) => {
  if (draggedId === targetId) return ids;
  const next = [...ids];
  const fromIndex = next.indexOf(draggedId);
  const toIndex = next.indexOf(targetId);
  if (fromIndex < 0 || toIndex < 0) return ids;
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  return next;
};

export function TrainingChecklist() {
  const {
    phases,
    loading,
    getPhaseProgress,
    toggleTask,
    createPhase,
    updatePhase,
    deletePhase,
    reorderPhases,
    createGroup,
    updateGroup,
    deleteGroup,
    reorderGroups,
    createTask,
    updateTask,
    deleteTask,
    reorderTasks,
  } = useTrainingStore();

  const [newPhaseTitle, setNewPhaseTitle] = useState("");
  const [newGroupByPhase, setNewGroupByPhase] = useState<Record<string, string>>({});
  const [newTaskByScope, setNewTaskByScope] = useState<Record<string, { label: string; param: string }>>({});
  const [editState, setEditState] = useState<EditState | null>(null);
  const [dragItem, setDragItem] = useState<DragItem | null>(null);

  const updateScopeDraft = (scopeKey: string, value: { label: string; param: string }) => {
    setNewTaskByScope((previous) => ({ ...previous, [scopeKey]: value }));
  };

  const clearScopeDraft = (scopeKey: string) => {
    setNewTaskByScope((previous) => ({ ...previous, [scopeKey]: { label: "", param: "" } }));
  };

  const handleCreatePhase = async () => {
    const value = newPhaseTitle.trim();
    if (!value) return;
    await createPhase(value);
    setNewPhaseTitle("");
  };

  const handleSaveEdit = async () => {
    if (!editState) return;
    const label = editState.label.trim();
    if (!label) return;

    if (editState.mode === "phase") {
      await updatePhase(editState.id, label);
    } else if (editState.mode === "group") {
      await updateGroup(editState.id, label);
    } else {
      await updateTask(editState.id, label, editState.param.trim());
    }

    setEditState(null);
  };

  const handlePhaseDrop = async (targetPhaseId: string) => {
    if (!dragItem || dragItem.kind !== "phase") return;
    const next = reorderIds(
      phases.map((phase) => phase.id),
      dragItem.phaseId,
      targetPhaseId,
    );
    setDragItem(null);
    if (next.join("|") === phases.map((phase) => phase.id).join("|")) return;
    await reorderPhases(next);
  };

  const handleGroupDrop = async (phaseId: string, targetGroupId: string) => {
    if (!dragItem || dragItem.kind !== "group" || dragItem.phaseId !== phaseId) return;
    const phase = phases.find((item) => item.id === phaseId);
    if (!phase) return;
    const currentIds = phase.groups.map((group) => group.id);
    const next = reorderIds(currentIds, dragItem.groupId, targetGroupId);
    setDragItem(null);
    if (next.join("|") === currentIds.join("|")) return;
    await reorderGroups(phaseId, next);
  };

  const handleTaskDrop = async (phaseId: string, groupId: string | null, targetTaskId: string) => {
    if (!dragItem || dragItem.kind !== "task") return;
    if (dragItem.phaseId !== phaseId || dragItem.groupId !== groupId) return;

    const phase = phases.find((item) => item.id === phaseId);
    if (!phase) return;
    const taskList = groupId ? phase.groups.find((group) => group.id === groupId)?.tasks ?? [] : phase.tasks;
    const currentIds = taskList.map((task) => task.id);
    const next = reorderIds(currentIds, dragItem.taskId, targetTaskId);
    setDragItem(null);
    if (next.join("|") === currentIds.join("|")) return;
    await reorderTasks(phaseId, groupId, next);
  };

  const handleDragStart = (event: DragEvent<HTMLElement>, item: DragItem) => {
    event.stopPropagation();
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", item.kind);
    setDragItem(item);
  };

  const handleDragOver = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "move";
  };

  const handleDragEnd = (event: DragEvent<HTMLElement>) => {
    event.stopPropagation();
    setDragItem(null);
  };

  return (
    <Card className="border p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h3 className="text-lg" style={{ color: "var(--primary)" }}>
          Checklist
        </h3>
      </div>

      <div className="mb-4 rounded-lg border border-dashed p-3">
        <p className="text-sm text-muted-foreground mb-2">Thêm giai đoạn mới</p>
        <div className="flex flex-col sm:flex-row gap-2">
          <input
            className="flex-1 rounded border bg-background px-3 py-2 text-sm"
            value={newPhaseTitle}
            onChange={(event) => setNewPhaseTitle(event.target.value)}
            placeholder="Ví dụ: Giai đoạn 6 — Calibration"
          />
          <Button onClick={() => void handleCreatePhase()} disabled={!newPhaseTitle.trim() || loading}>
            <Plus className="w-4 h-4 mr-2" />
            Thêm giai đoạn mới
          </Button>
        </div>
      </div>

      {editState && (
        <div className="mb-4 rounded-lg border p-3 bg-background-secondary/60">
          <p className="text-sm text-muted-foreground mb-2">Đang chỉnh sửa</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <input
              className="rounded border bg-background px-3 py-2 text-sm"
              value={editState.label}
              onChange={(event) => setEditState((prev) => (prev ? { ...prev, label: event.target.value } : prev))}
            />
            {editState.mode === "task" && (
              <input
                className="rounded border bg-background px-3 py-2 text-sm"
                value={editState.param}
                onChange={(event) => setEditState((prev) => (prev ? { ...prev, param: event.target.value } : prev))}
                placeholder="param (optional)"
              />
            )}
          </div>
          <div className="mt-2 flex gap-2">
            <Button size="sm" onClick={() => void handleSaveEdit()}>
              Lưu
            </Button>
            <Button size="sm" variant="outline" onClick={() => setEditState(null)}>
              Huỷ
            </Button>
          </div>
        </div>
      )}

      <Accordion type="multiple" className="w-full">
        {phases.map((phase) => {
          const progress = getPhaseProgress(phase.id);
          const directScopeKey = `${phase.id}::__root__`;
          const directDraft = newTaskByScope[directScopeKey] ?? { label: "", param: "" };

          return (
            <AccordionItem
              value={phase.id}
              key={phase.id}
              draggable
              onDragStart={(event) => handleDragStart(event, { kind: "phase", phaseId: phase.id })}
              onDragOver={handleDragOver}
              onDrop={(event) => {
                event.stopPropagation();
                void handlePhaseDrop(phase.id);
              }}
              onDragEnd={handleDragEnd}
            >
              <AccordionTrigger className="hover:no-underline">
                <div className="flex w-full flex-wrap items-start justify-between gap-3 pr-2">
                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    <span className="text-muted-foreground"><GripVertical className="w-4 h-4" /></span>
                    <span className="text-sm font-medium text-foreground">{phase.title}</span>
                    <Badge className={STATUS_CLASS[progress.status]}>{STATUS_TEXT[progress.status]}</Badge>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {progress.checkedCount}/{progress.totalCount}
                  </span>
                </div>
              </AccordionTrigger>

              <AccordionContent>
                <div className="mb-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setEditState({ mode: "phase", id: phase.id, phaseId: phase.id, groupId: null, label: phase.title, param: "" })}
                  >
                    Sửa giai đoạn
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => void deletePhase(phase.id)}>
                    <Trash2 className="w-4 h-4 mr-1" /> Xoá giai đoạn
                  </Button>
                </div>

                <div className="space-y-4">
                  {phase.groups.map((group) => {
                    const groupScopeKey = `${phase.id}:${group.id}`;
                    const groupDraft = newTaskByScope[groupScopeKey] ?? { label: "", param: "" };

                    return (
                      <div
                        key={group.id}
                        className="rounded-lg border border-border/70 p-3"
                        draggable
                        onDragStart={(event) =>
                          handleDragStart(event, { kind: "group", phaseId: phase.id, groupId: group.id })
                        }
                        onDragOver={handleDragOver}
                        onDrop={(event) => {
                          event.stopPropagation();
                          void handleGroupDrop(phase.id, group.id);
                        }}
                        onDragEnd={handleDragEnd}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                          <p className="text-sm flex items-center gap-2" style={{ color: "var(--primary)" }}>
                            <span className="text-muted-foreground"><GripVertical className="w-4 h-4" /></span>
                            {group.title}
                          </p>
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() =>
                                setEditState({
                                  mode: "group",
                                  id: group.id,
                                  phaseId: phase.id,
                                  groupId: group.id,
                                  label: group.title,
                                  param: "",
                                })
                              }
                            >
                              Sửa
                            </Button>
                            <Button size="sm" variant="outline" onClick={() => void deleteGroup(group.id)}>
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>

                        <div className="space-y-2">
                          {group.tasks.map((task) => (
                            <div
                              key={task.id}
                              className="flex flex-col gap-2 rounded-md px-2 py-1 hover:bg-background-secondary"
                              draggable
                              onDragStart={(event) =>
                                handleDragStart(event, {
                                  kind: "task",
                                  phaseId: phase.id,
                                  groupId: group.id,
                                  taskId: task.id,
                                })
                              }
                              onDragOver={handleDragOver}
                              onDrop={(event) => {
                                event.stopPropagation();
                                void handleTaskDrop(phase.id, group.id, task.id);
                              }}
                              onDragEnd={handleDragEnd}
                            >
                              <label className="flex cursor-pointer items-start gap-2">
                                <span className="text-muted-foreground mt-0.5"><GripVertical className="w-4 h-4" /></span>
                                <Checkbox
                                  checked={task.checked}
                                  onCheckedChange={(checked) => void toggleTask(task.id, checked === true)}
                                  className="mt-0.5"
                                />
                                <span className="text-sm text-foreground leading-relaxed">{renderTaskLabel(task)}</span>
                              </label>
                              <div className="ml-6 flex flex-wrap gap-1">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() =>
                                    setEditState({
                                      mode: "task",
                                      id: task.id,
                                      phaseId: phase.id,
                                      groupId: group.id,
                                      label: task.label,
                                      param: task.param ?? "",
                                    })
                                  }
                                >
                                  Sửa task
                                </Button>
                                <Button size="sm" variant="outline" onClick={() => void deleteTask(task.id)}>
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>

                        <div className="mt-3 grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-2">
                          <input
                            className="rounded border bg-background px-3 py-2 text-sm"
                            value={groupDraft.label}
                            onChange={(event) =>
                              updateScopeDraft(groupScopeKey, {
                                ...groupDraft,
                                label: event.target.value,
                              })
                            }
                            placeholder="Task mới"
                          />
                          <input
                            className="rounded border bg-background px-3 py-2 text-sm"
                            value={groupDraft.param}
                            onChange={(event) =>
                              updateScopeDraft(groupScopeKey, {
                                ...groupDraft,
                                param: event.target.value,
                              })
                            }
                            placeholder="Param (optional)"
                          />
                          <Button
                            onClick={() =>
                              void createTask(phase.id, group.id, groupDraft.label, groupDraft.param).then(() =>
                                clearScopeDraft(groupScopeKey),
                              )
                            }
                            disabled={!groupDraft.label.trim()}
                          >
                            <Plus className="w-4 h-4 mr-1" />
                            Thêm task
                          </Button>
                        </div>
                      </div>
                    );
                  })}

                  {phase.tasks.length > 0 && (
                    <div className="rounded-lg border border-border/70 p-3">
                      <p className="text-sm mb-2" style={{ color: "var(--primary)" }}>
                        Tasks trực tiếp
                      </p>
                      <div className="space-y-2">
                        {phase.tasks.map((task) => (
                          <div
                            key={task.id}
                            className="flex flex-col gap-2 rounded-md px-2 py-1 hover:bg-background-secondary"
                            draggable
                            onDragStart={(event) =>
                              handleDragStart(event, {
                                kind: "task",
                                phaseId: phase.id,
                                groupId: null,
                                taskId: task.id,
                              })
                            }
                            onDragOver={handleDragOver}
                            onDrop={(event) => {
                              event.stopPropagation();
                              void handleTaskDrop(phase.id, null, task.id);
                            }}
                            onDragEnd={handleDragEnd}
                          >
                            <label className="flex cursor-pointer items-start gap-2">
                              <span className="text-muted-foreground mt-0.5"><GripVertical className="w-4 h-4" /></span>
                              <Checkbox
                                checked={task.checked}
                                onCheckedChange={(checked) => void toggleTask(task.id, checked === true)}
                                className="mt-0.5"
                              />
                              <span className="text-sm text-foreground leading-relaxed">{renderTaskLabel(task)}</span>
                            </label>
                            <div className="ml-6 flex flex-wrap gap-1">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() =>
                                  setEditState({
                                    mode: "task",
                                    id: task.id,
                                    phaseId: phase.id,
                                    groupId: null,
                                    label: task.label,
                                    param: task.param ?? "",
                                  })
                                }
                              >
                                Sửa task
                              </Button>
                              <Button size="sm" variant="outline" onClick={() => void deleteTask(task.id)}>
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="rounded-lg border border-dashed p-3">
                    <p className="text-sm text-muted-foreground mb-2">Thêm subgroup vào giai đoạn</p>
                    <div className="flex flex-col sm:flex-row gap-2">
                      <input
                        className="flex-1 rounded border bg-background px-3 py-2 text-sm"
                        value={newGroupByPhase[phase.id] ?? ""}
                        onChange={(event) =>
                          setNewGroupByPhase((previous) => ({
                            ...previous,
                            [phase.id]: event.target.value,
                          }))
                        }
                        placeholder="Ví dụ: 2.1 Sample curation"
                      />
                      <Button
                        onClick={() =>
                          void createGroup(phase.id, (newGroupByPhase[phase.id] ?? "").trim()).then(() =>
                            setNewGroupByPhase((previous) => ({ ...previous, [phase.id]: "" })),
                          )
                        }
                        disabled={!((newGroupByPhase[phase.id] ?? "").trim())}
                      >
                        <Plus className="w-4 h-4 mr-1" />
                        Thêm subgroup
                      </Button>
                    </div>
                  </div>

                  <div className="rounded-lg border border-dashed p-3">
                    <p className="text-sm text-muted-foreground mb-2">Thêm task trực tiếp cho giai đoạn</p>
                    <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-2">
                      <input
                        className="rounded border bg-background px-3 py-2 text-sm"
                        value={directDraft.label}
                        onChange={(event) =>
                          updateScopeDraft(directScopeKey, {
                            ...directDraft,
                            label: event.target.value,
                          })
                        }
                        placeholder="Task mới"
                      />
                      <input
                        className="rounded border bg-background px-3 py-2 text-sm"
                        value={directDraft.param}
                        onChange={(event) =>
                          updateScopeDraft(directScopeKey, {
                            ...directDraft,
                            param: event.target.value,
                          })
                        }
                        placeholder="Param (optional)"
                      />
                      <Button
                        onClick={() =>
                          void createTask(phase.id, null, directDraft.label, directDraft.param).then(() =>
                            clearScopeDraft(directScopeKey),
                          )
                        }
                        disabled={!directDraft.label.trim()}
                      >
                        <Plus className="w-4 h-4 mr-1" />
                        Thêm task
                      </Button>
                    </div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          );
        })}
      </Accordion>
    </Card>
  );
}
