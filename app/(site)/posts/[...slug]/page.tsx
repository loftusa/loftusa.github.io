import "katex/dist/katex.min.css";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getAllPostSlugs, getPostBySlug } from "@/lib/content";
import { formatDate } from "@/lib/date";
import Disqus from "@/components/Disqus";
import styles from "./post.module.css";

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllPostSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const post = await getPostBySlug(slug);
  if (!post) return {};
  return {
    title: post.title,
    description: post.excerpt,
    alternates: { canonical: post.permalink },
  };
}

export default async function PostPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const post = await getPostBySlug(slug);
  if (!post) notFound();

  return (
    <main className="page-main reading">
      <article>
        <p className={styles.eyebrow}>{formatDate(post.date)}</p>
        <h1 className={styles.title}>{post.title}</h1>
        <div className="prose" dangerouslySetInnerHTML={{ __html: post.html }} />
      </article>
      {post.comments && <Disqus identifier={post.permalink} title={post.title} />}
    </main>
  );
}
