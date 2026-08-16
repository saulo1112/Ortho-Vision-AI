import * as Haptics from 'expo-haptics';
import { Image } from 'expo-image';
import { router } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from 'react-native';
import Swipeable, { type SwipeableMethods } from 'react-native-gesture-handler/ReanimatedSwipeable';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, { type SharedValue, useAnimatedStyle } from 'react-native-reanimated';

import { deleteInference, listInferences } from '../api/client';
import type { InferenceSummary } from '../api/types';
import { ChevronLeftIcon, TrashIcon } from '../components/icons';
import { percent, timeAgo } from '../lib/format';
import { classStyle } from '../theme/classes';
import { colors, fonts, radius, spacing } from '../theme/tokens';

export default function HistoryScreen() {
  const [items, setItems] = useState<InferenceSummary[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setError(null);
      const res = await listInferences(50);
      setItems(res.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load history');
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleDelete = useCallback(
    async (id: string) => {
      // Optimistic: remove locally first so the swipe feels instant, like
      // deleting a photo. If the backend call fails, put it back and surface
      // the error via a fresh load.
      const removed = items.find((i) => i.inference_id === id);
      setItems((prev) => prev.filter((i) => i.inference_id !== id));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      try {
        await deleteInference(id);
      } catch (err) {
        if (removed) setItems((prev) => [...prev, removed].sort(byCreatedAtDesc));
        setError(err instanceof Error ? err.message : 'Could not delete');
      }
    },
    [items],
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
      <View style={styles.headerRow}>
        <Pressable
          onPress={() => router.back()}
          style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
          hitSlop={8}
        >
          <ChevronLeftIcon color={colors.textSecondary} />
        </Pressable>
        <Text style={styles.title}>History</Text>
        <View style={styles.headerSpacer} />
      </View>

      <FlatList
        data={items}
        keyExtractor={(item) => item.inference_id}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            tintColor={colors.primary}
            onRefresh={() => {
              setRefreshing(true);
              load();
            }}
          />
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>{error ? 'Connection error' : 'No analyses yet'}</Text>
            <Text style={styles.emptyCopy}>
              {error ?? 'Radiographs you analyze will appear here.'}
            </Text>
          </View>
        }
        renderItem={({ item }) => <HistoryRow item={item} onDelete={handleDelete} />}
      />
    </SafeAreaView>
  );
}

function byCreatedAtDesc(a: InferenceSummary, b: InferenceSummary) {
  return b.created_at.localeCompare(a.created_at);
}

function HistoryRow({
  item,
  onDelete,
}: {
  item: InferenceSummary;
  onDelete: (id: string) => void;
}) {
  const swipeableRef = useRef<SwipeableMethods>(null);
  const classNames = Object.keys(item.counts);
  const label =
    classNames.length > 0
      ? classNames.map((name) => classStyle(name).label).join(' · ')
      : 'No detections';
  const dotColor = classNames.length > 0 ? classStyle(classNames[0]).color : colors.textTertiary;

  const renderRightActions = useCallback(
    (progress: SharedValue<number>) => {
      const style = useAnimatedStyle(() => ({
        transform: [{ scale: Math.min(progress.value, 1) }],
      }));
      return (
        <Animated.View style={[styles.deleteAction, style]}>
          <Pressable
            onPress={() => {
              swipeableRef.current?.close();
              onDelete(item.inference_id);
            }}
            style={({ pressed }) => [styles.deleteButton, pressed && styles.pressed]}
            hitSlop={8}
          >
            <TrashIcon color="#FFFFFF" size={20} />
          </Pressable>
        </Animated.View>
      );
    },
    [item.inference_id, onDelete],
  );

  return (
    <Swipeable
      ref={swipeableRef}
      renderRightActions={renderRightActions}
      rightThreshold={44}
      friction={1.7}
      overshootRight={false}
    >
      <Pressable
        onPress={() => router.push(`/inference/${item.inference_id}`)}
        style={({ pressed }) => [styles.row, pressed && styles.pressed]}
      >
        {item.thumbnail_b64 ? (
          <Image
            source={{ uri: `data:image/jpeg;base64,${item.thumbnail_b64}` }}
            style={styles.thumb}
            contentFit="cover"
          />
        ) : (
          <View style={[styles.thumb, styles.thumbPlaceholder]} />
        )}
        <View style={styles.rowBody}>
          <View style={styles.rowTitleLine}>
            <View style={[styles.dot, { backgroundColor: dotColor }]} />
            <Text style={styles.rowTitle} numberOfLines={1}>
              {label}
            </Text>
          </View>
          <Text style={styles.rowMeta}>
            {timeAgo(item.created_at)}
            {item.max_confidence != null ? `  ·  ${percent(item.max_confidence)} confidence` : ''}
          </Text>
        </View>
      </Pressable>
    </Swipeable>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  title: { color: colors.text, fontSize: 18, fontWeight: '700', fontFamily: fonts?.sans },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerSpacer: { width: 42, height: 42 },
  pressed: { opacity: 0.65 },
  list: { padding: spacing.lg, gap: spacing.sm, flexGrow: 1 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.sm,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
  },
  thumb: { width: 58, height: 58, borderRadius: 12 },
  thumbPlaceholder: { backgroundColor: colors.bgElevated },
  rowBody: { flex: 1, gap: 3 },
  rowTitleLine: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  rowTitle: { color: colors.text, fontSize: 14, fontWeight: '600', flex: 1, fontFamily: fonts?.sans },
  rowMeta: { color: colors.textTertiary, fontSize: 12, fontFamily: fonts?.sans },
  empty: { alignItems: 'center', gap: spacing.sm, marginTop: 80 },
  emptyTitle: { color: colors.text, fontSize: 16, fontWeight: '600', fontFamily: fonts?.sans },
  emptyCopy: { color: colors.textSecondary, fontSize: 13, textAlign: 'center', fontFamily: fonts?.sans },
  deleteAction: {
    justifyContent: 'center',
    alignItems: 'center',
    width: 72,
  },
  deleteButton: {
    flex: 1,
    alignSelf: 'stretch',
    marginLeft: spacing.sm,
    backgroundColor: colors.danger,
    borderRadius: radius.control,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
