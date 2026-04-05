package service

import (
	"encoding/json"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
)

const compatPromptCacheKeyPrefix = "compat_cc_"
const responsesPromptCacheKeyPrefix = "compat_rsp_"

func shouldAutoInjectPromptCacheKeyForCompat(model string) bool {
	switch normalizeCodexModel(strings.TrimSpace(model)) {
	case "gpt-5.4", "gpt-5.3-codex":
		return true
	default:
		return false
	}
}

func deriveCompatPromptCacheKey(req *apicompat.ChatCompletionsRequest, mappedModel string) string {
	if req == nil {
		return ""
	}

	normalizedModel := normalizeCodexModel(strings.TrimSpace(mappedModel))
	if normalizedModel == "" {
		normalizedModel = normalizeCodexModel(strings.TrimSpace(req.Model))
	}
	if normalizedModel == "" {
		normalizedModel = strings.TrimSpace(req.Model)
	}

	seedParts := []string{"model=" + normalizedModel}
	if req.ReasoningEffort != "" {
		seedParts = append(seedParts, "reasoning_effort="+strings.TrimSpace(req.ReasoningEffort))
	}
	if len(req.ToolChoice) > 0 {
		seedParts = append(seedParts, "tool_choice="+normalizeCompatSeedJSON(req.ToolChoice))
	}
	if len(req.Tools) > 0 {
		if raw, err := json.Marshal(req.Tools); err == nil {
			seedParts = append(seedParts, "tools="+normalizeCompatSeedJSON(raw))
		}
	}
	if len(req.Functions) > 0 {
		if raw, err := json.Marshal(req.Functions); err == nil {
			seedParts = append(seedParts, "functions="+normalizeCompatSeedJSON(raw))
		}
	}

	firstUserCaptured := false
	for _, msg := range req.Messages {
		switch strings.TrimSpace(msg.Role) {
		case "system":
			seedParts = append(seedParts, "system="+normalizeCompatSeedJSON(msg.Content))
		case "user":
			if !firstUserCaptured {
				seedParts = append(seedParts, "first_user="+normalizeCompatSeedJSON(msg.Content))
				firstUserCaptured = true
			}
		}
	}

	return compatPromptCacheKeyPrefix + hashSensitiveValueForLog(strings.Join(seedParts, "|"))
}

func deriveResponsesPromptCacheKey(req *apicompat.ResponsesRequest, mappedModel string) string {
	if req == nil {
		return ""
	}

	normalizedModel := resolveOpenAIUpstreamModel(strings.TrimSpace(mappedModel))
	if normalizedModel == "" {
		normalizedModel = resolveOpenAIUpstreamModel(strings.TrimSpace(req.Model))
	}
	if normalizedModel == "" {
		normalizedModel = strings.TrimSpace(req.Model)
	}

	seedParts := []string{"model=" + normalizedModel}
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		seedParts = append(seedParts, "reasoning_effort="+strings.TrimSpace(req.Reasoning.Effort))
	}
	if len(req.ToolChoice) > 0 {
		seedParts = append(seedParts, "tool_choice="+normalizeCompatSeedJSON(req.ToolChoice))
	}
	if len(req.Tools) > 0 {
		if raw, err := json.Marshal(req.Tools); err == nil {
			seedParts = append(seedParts, "tools="+normalizeCompatSeedJSON(raw))
		}
	}
	if instructions := normalizeResponsesStringSeed(strings.TrimSpace(extractResponsesInstructionsForSeed(req))); instructions != "" {
		seedParts = append(seedParts, "system="+instructions)
	}

	systemParts, firstUser := extractResponsesInputSeedParts(req.Input)
	for _, part := range systemParts {
		seedParts = append(seedParts, "system="+part)
	}
	if firstUser != "" {
		seedParts = append(seedParts, "first_user="+firstUser)
	}

	return responsesPromptCacheKeyPrefix + hashSensitiveValueForLog(strings.Join(seedParts, "|"))
}

func deriveResponsesPromptCacheKeyFromBody(body []byte, mappedModel string) string {
	if len(body) == 0 {
		return ""
	}

	var req apicompat.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return deriveResponsesPromptCacheKey(&req, mappedModel)
}

func extractResponsesInstructionsForSeed(req *apicompat.ResponsesRequest) string {
	if req == nil {
		return ""
	}
	return req.Instructions
}

func extractResponsesInputSeedParts(raw json.RawMessage) ([]string, string) {
	if len(raw) == 0 {
		return nil, ""
	}

	var inputString string
	if err := json.Unmarshal(raw, &inputString); err == nil {
		return nil, normalizeResponsesStringSeed(strings.TrimSpace(inputString))
	}

	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, ""
	}

	systemParts := make([]string, 0)
	implicitUserItems := make([]json.RawMessage, 0)
	firstUser := ""

	for _, itemRaw := range items {
		itemRaw = trimJSONRawMessage(itemRaw)
		if len(itemRaw) == 0 {
			continue
		}

		var item map[string]json.RawMessage
		if err := json.Unmarshal(itemRaw, &item); err != nil {
			if firstUser == "" && len(implicitUserItems) == 0 {
				implicitUserItems = append(implicitUserItems, itemRaw)
			}
			continue
		}

		role := rawJSONStringField(item["role"])
		contentRaw := trimJSONRawMessage(item["content"])

		switch role {
		case "system":
			if normalized := normalizeResponsesSeedRaw(contentRaw, itemRaw); normalized != "" {
				systemParts = append(systemParts, normalized)
			}
			continue
		case "user":
			if firstUser == "" {
				firstUser = normalizeResponsesSeedRaw(contentRaw, itemRaw)
			}
			implicitUserItems = implicitUserItems[:0]
			continue
		}

		if firstUser != "" {
			continue
		}

		if isImplicitResponsesUserItem(item) {
			implicitUserItems = append(implicitUserItems, itemRaw)
			continue
		}

		if len(implicitUserItems) > 0 {
			break
		}
	}

	if firstUser == "" && len(implicitUserItems) > 0 {
		if rawItems, err := json.Marshal(implicitUserItems); err == nil {
			firstUser = normalizeCompatSeedJSON(rawItems)
		}
	}

	return systemParts, firstUser
}

func normalizeResponsesStringSeed(v string) string {
	if strings.TrimSpace(v) == "" {
		return ""
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return normalizeCompatSeedJSON(raw)
}

func normalizeResponsesSeedRaw(primary json.RawMessage, fallback json.RawMessage) string {
	if normalized := normalizeCompatSeedJSON(trimJSONRawMessage(primary)); normalized != "" {
		return normalized
	}
	return normalizeCompatSeedJSON(trimJSONRawMessage(fallback))
}

func rawJSONStringField(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return strings.TrimSpace(s)
}

func isImplicitResponsesUserItem(item map[string]json.RawMessage) bool {
	typ := rawJSONStringField(item["type"])
	switch typ {
	case "input_text", "input_image", "text", "message":
		return true
	case "function_call", "function_call_output", "reasoning", "item_reference":
		return false
	}

	if len(trimJSONRawMessage(item["content"])) > 0 {
		return true
	}
	if len(trimJSONRawMessage(item["text"])) > 0 {
		return true
	}
	if len(trimJSONRawMessage(item["image_url"])) > 0 {
		return true
	}

	return false
}

func trimJSONRawMessage(raw json.RawMessage) json.RawMessage {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return nil
	}
	return json.RawMessage(trimmed)
}

func normalizeCompatSeedJSON(v json.RawMessage) string {
	if len(v) == 0 {
		return ""
	}
	var tmp any
	if err := json.Unmarshal(v, &tmp); err != nil {
		return string(v)
	}
	out, err := json.Marshal(tmp)
	if err != nil {
		return string(v)
	}
	return string(out)
}
